from __future__ import annotations

from pathlib import Path
from datetime import datetime
import argparse
import csv
import json
import shutil
import textwrap

from PIL import Image, ImageDraw, ImageOps


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "fabricDD.v2i.yolov8"
REPORT_DIR = PROJECT_DIR / "reports" / "cross_group_related_20261008_114158_369379"
CANDIDATES = REPORT_DIR / "cross_group_similarity_candidates.csv"
REVIEW = REPORT_DIR / "cross_group_review.csv"
SUMMARY = REPORT_DIR / "cross_group_review_coverage_summary.txt"
CHECKPOINT = REPORT_DIR / "cross_group_review_resume_checkpoint.json"
CONTINUE_DIR = REPORT_DIR / "review_sheets_continue"
CONSOLIDATION = REPORT_DIR / "cross_group_consolidation_map.csv"

FIELDS = [
    "review_id",
    "left_image",
    "right_image",
    "left_group_id",
    "right_group_id",
    "left_source_type",
    "right_source_type",
    "hash_distance",
    "proposed_relationship",
    "visual_evidence",
    "decision",
    "unresolved_issues",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def endpoint_key(row: dict[str, str]) -> tuple[str, str]:
    return tuple(sorted((row["left_image"].replace("\\", "/"), row["right_image"].replace("\\", "/"))))


def candidate_key(row: dict[str, str]) -> tuple[str, str]:
    return tuple(sorted((row["left_image"].replace("\\", "/"), row["right_image"].replace("\\", "/"))))


def is_cross_split(row: dict[str, str]) -> bool:
    return row["left_split"] != row["right_split"]


def candidate_with_index(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out = []
    for index, row in enumerate(rows, 1):
        out.append({"candidate_index_1based": str(index), **row})
    return out


def reviewed_keys(review_rows: list[dict[str, str]]) -> set[tuple[str, str]]:
    return {endpoint_key(row) for row in review_rows}


def consolidated_keys() -> set[tuple[str, str]]:
    if not CONSOLIDATION.exists():
        return set()
    rows = read_csv(CONSOLIDATION)
    keys = set()
    for row in rows:
        left = row.get("left_image", "").replace("\\", "/")
        right = row.get("right_image", "").replace("\\", "/")
        if left and right:
            keys.add(tuple(sorted((left, right))))
    return keys


def pending_cross_split(candidates: list[dict[str, str]], review_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    done = reviewed_keys(review_rows) | consolidated_keys()
    return [row for row in candidate_with_index(candidates) if is_cross_split(row) and candidate_key(row) not in done]


def pending_within_split(candidates: list[dict[str, str]], review_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    done = reviewed_keys(review_rows) | consolidated_keys()
    return [row for row in candidate_with_index(candidates) if not is_cross_split(row) and candidate_key(row) not in done]


def pending_by_mode(candidates: list[dict[str, str]], review_rows: list[dict[str, str]], mode: str) -> list[dict[str, str]]:
    if mode == "cross":
        return pending_cross_split(candidates, review_rows)
    if mode == "within":
        return pending_within_split(candidates, review_rows)
    if mode == "auto":
        cross = pending_cross_split(candidates, review_rows)
        return cross if cross else pending_within_split(candidates, review_rows)
    raise ValueError(f"unknown mode {mode!r}")


def make_review_row(row: dict[str, str], evidence: str, decision: str, unresolved: str = "") -> dict[str, str]:
    idx = int(row["candidate_index_1based"])
    return {
        "review_id": f"X{idx:04d}",
        "left_image": row["left_image"],
        "right_image": row["right_image"],
        "left_group_id": row["left_group_id"],
        "right_group_id": row["right_group_id"],
        "left_source_type": row["left_source_type"],
        "right_source_type": row["right_source_type"],
        "hash_distance": row["hash_distance"],
        "proposed_relationship": "possible transformed duplicate/source-photo family candidate",
        "visual_evidence": evidence,
        "decision": decision,
        "unresolved_issues": unresolved,
    }


def review_sort_key(row: dict[str, str]) -> tuple[int, str]:
    review_id = row["review_id"]
    if review_id.startswith("X") and review_id[1:].isdigit():
        return (int(review_id[1:]), review_id)
    return (10_000_000, review_id)


def append_batch(batch_num: int, decisions_path: Path, mode: str = "cross") -> None:
    candidates = read_csv(CANDIDATES)
    review_rows = read_csv(REVIEW)
    pending = {row["candidate_index_1based"]: row for row in pending_by_mode(candidates, review_rows, mode)}
    decisions = read_csv(decisions_path)
    existing = reviewed_keys(review_rows)
    new_rows = []
    for decision in decisions:
        idx = decision["candidate_index_1based"]
        if idx not in pending:
            raise SystemExit(f"candidate {idx} is not pending in mode {mode}")
        row = pending[idx]
        key = candidate_key(row)
        if key in existing:
            raise SystemExit(f"candidate {idx} already reviewed")
        new_rows.append(make_review_row(row, decision["visual_evidence"], decision["decision"], decision.get("unresolved_issues", "")))
    shutil.copy2(REVIEW, REPORT_DIR / f"cross_group_review.backup_before_batch_{batch_num:03d}_{datetime.now():%Y%m%d_%H%M%S}.csv")
    review_rows.extend(new_rows)
    review_rows.sort(key=review_sort_key)
    write_csv(REVIEW, review_rows, FIELDS)
    write_summary(batch_num, [int(row["candidate_index_1based"]) for row in decisions])


def write_summary(batch_num: int, batch_indices: list[int]) -> None:
    candidates = read_csv(CANDIDATES)
    review_rows = read_csv(REVIEW)
    total = len(candidates)
    special = sum(1 for row in review_rows if row["decision"] == "different_photos_unresolved_relationship_retained")
    reviewed = len(review_rows) - special
    consolidated = len(consolidated_keys())
    pending = total - reviewed - consolidated
    pending_x = len(pending_cross_split(candidates, review_rows))
    pending_w = len(pending_within_split(candidates, review_rows))
    counts = {}
    for row in review_rows:
        counts[row["decision"]] = counts.get(row["decision"], 0) + 1

    lines = [
        "AI-assisted cross-group related-source-photo review coverage summary",
        f"Updated: {datetime.now():%Y-%m-%d %H:%M:%S} after Batch {batch_num}",
        "",
        f"Total candidate pairs: {total}",
        f"Visually reviewed candidate pairs: {reviewed}",
        f"Consolidated candidate pairs: {consolidated}",
        f"Pending candidate pairs: {pending}",
        f"Pending cross-split pairs: {pending_x}",
        f"Pending within-split pairs: {pending_w}",
        f"Special C00680 unresolved rows: {special}",
        f"Coverage check: {reviewed} + {consolidated} + {pending} = {total}",
        "",
        "Decision counts including special rows:",
    ]
    for decision, count in sorted(counts.items()):
        lines.append(f"- {decision}: {count}")
    lines.extend(["", f"Batch {batch_num} audit:"])
    ids = {
        int(row["review_id"][1:]): row
        for row in review_rows
        if row["review_id"].startswith("X") and row["review_id"][1:].isdigit()
    }
    for idx in batch_indices:
        row = ids.get(idx)
        lines.append(f"- candidate {idx}: {1 if row else 0} matching review row(s) {row['review_id'] if row else 'MISSING'}")
    lines.extend([
        "",
        f"Active report: {REVIEW}",
        f"Consolidation map: {REPORT_DIR / 'cross_group_consolidation_map.csv'}",
        f"Confirmed family mapping: {REPORT_DIR / 'confirmed_source_photo_families.csv'}",
        f"Unresolved relationships: {REPORT_DIR / 'cross_group_unresolved_relationships.csv'}",
        f"Next resume: Batch {batch_num + 1} from first pending cross-split pairs, then within-split pairs, by candidate order using normalized endpoint keys.",
    ])
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    CHECKPOINT.write_text(json.dumps({
        "updated": f"{datetime.now():%Y-%m-%d %H:%M:%S} after Batch {batch_num}",
        "total": total,
        "reviewed": reviewed,
        "consolidated": consolidated,
        "pending": pending,
        "pending_cross_split": pending_x,
        "pending_within_split": pending_w,
        "special_c00680_rows": special,
        "next_batch": batch_num + 1,
        "active_report": str(REVIEW),
        "summary": str(SUMMARY),
    }, indent=2), encoding="utf-8")


def make_sheet(batch_num: int, batch_size: int = 20, mode: str = "cross") -> None:
    candidates = read_csv(CANDIDATES)
    review_rows = read_csv(REVIEW)
    selected = pending_by_mode(candidates, review_rows, mode)[:batch_size]
    if not selected:
        print(f"No pending {mode} candidates.")
        return
    CONTINUE_DIR.mkdir(exist_ok=True)
    tag = "cross_split" if mode in ("cross", "auto") and any(is_cross_split(row) for row in selected) else "within_split"
    manifest = CONTINUE_DIR / f"batch_{tag}_{batch_num:03d}_manifest.csv"
    write_csv(manifest, selected, list(selected[0].keys()))

    cell_w = 540
    cell_h = 410
    header_h = 90
    pair_h = header_h + cell_h
    per_page = 2
    for start in range(0, len(selected), per_page):
        batch = selected[start:start + per_page]
        page = start // per_page + 1
        sheet = Image.new("RGB", (cell_w * 2, pair_h * len(batch)), "white")
        draw = ImageDraw.Draw(sheet)
        for offset, row in enumerate(batch):
            y0 = offset * pair_h
            title = (
                f"Candidate {row['candidate_index_1based']} | hash={row['hash_distance']} "
                f"color={row['color_distance']} | "
                f"{row['left_group_id'] or 'UNG'} <-> {row['right_group_id'] or 'UNG'}"
            )
            draw.text((10, y0 + 8), title, fill="black")
            for side, x0 in (("left", 0), ("right", cell_w)):
                image_rel = row[f"{side}_image"]
                label = (
                    f"{side[0].upper()}: {row[f'{side}_split']} | "
                    f"{row[f'{side}_group_id'] or 'UNGROUPED'} {image_rel} "
                    f"transform={row[f'{side}_transform']}"
                )
                draw.multiline_text((x0 + 10, y0 + 32), "\n".join(textwrap.wrap(label, width=80)), fill="black", spacing=3)
                draw.rectangle((x0 + 4, y0 + header_h, x0 + cell_w - 4, y0 + pair_h - 4), outline="#999999")
                try:
                    with Image.open(DATASET_DIR / Path(image_rel)) as image:
                        preview = ImageOps.contain(ImageOps.exif_transpose(image).convert("RGB"), (cell_w - 24, cell_h - 24))
                    px = x0 + (cell_w - preview.width) // 2
                    py = y0 + header_h + (cell_h - preview.height) // 2
                    sheet.paste(preview, (px, py))
                except (OSError, ValueError) as error:
                    draw.text((x0 + 10, y0 + header_h + 20), str(error), fill="red")
        sheet.save(CONTINUE_DIR / f"batch_{tag}_{batch_num:03d}_page_{page:02d}.jpg", quality=92)
    print(f"Prepared Batch {batch_num}: {len(selected)} candidates")


def state() -> None:
    candidates = read_csv(CANDIDATES)
    review_rows = read_csv(REVIEW)
    special = sum(1 for row in review_rows if row["decision"] == "different_photos_unresolved_relationship_retained")
    reviewed = len(review_rows) - special
    consolidated = len(consolidated_keys())
    pending = len(candidates) - reviewed - consolidated
    print(json.dumps({
        "total": len(candidates),
        "reviewed": reviewed,
        "consolidated": consolidated,
        "pending": pending,
        "pending_cross_split": len(pending_cross_split(candidates, review_rows)),
        "pending_within_split": len(pending_within_split(candidates, review_rows)),
        "special_c00680_rows": special,
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("state")
    make = sub.add_parser("make-sheet")
    make.add_argument("--batch", type=int, required=True)
    make.add_argument("--size", type=int, default=20)
    make.add_argument("--mode", choices=["cross", "within", "auto"], default="cross")
    append = sub.add_parser("append")
    append.add_argument("--batch", type=int, required=True)
    append.add_argument("--decisions", type=Path, required=True)
    append.add_argument("--mode", choices=["cross", "within", "auto"], default="cross")
    summarize = sub.add_parser("summarize")
    summarize.add_argument("--batch", type=int, required=True)
    summarize.add_argument("--decisions", type=Path, required=True)
    args = parser.parse_args()
    if args.cmd == "state":
        state()
    elif args.cmd == "make-sheet":
        make_sheet(args.batch, args.size, args.mode)
    elif args.cmd == "append":
        append_batch(args.batch, args.decisions, args.mode)
    elif args.cmd == "summarize":
        decisions = read_csv(args.decisions)
        write_summary(args.batch, [int(row["candidate_index_1based"]) for row in decisions])


if __name__ == "__main__":
    main()
