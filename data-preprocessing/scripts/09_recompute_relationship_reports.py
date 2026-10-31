from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
import csv
import shutil


PROJECT_DIR = Path(__file__).resolve().parent.parent
REPORT_DIR = PROJECT_DIR / "reports" / "cross_group_related_20261008_114158_369379"
CANDIDATES = REPORT_DIR / "cross_group_similarity_candidates.csv"
REVIEW = REPORT_DIR / "cross_group_review.csv"
CONSOLIDATION = REPORT_DIR / "cross_group_consolidation_map.csv"
FAMILIES = REPORT_DIR / "confirmed_source_photo_families.csv"
CONFLICTS = REPORT_DIR / "cross_group_conflict_report.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def pair_key(row: dict[str, str]) -> tuple[str, str]:
    return tuple(sorted((row["left_image"].replace("\\", "/"), row["right_image"].replace("\\", "/"))))


def source_unit(row: dict[str, str], side: str) -> str:
    group_id = row.get(f"{side}_group_id", "")
    if group_id:
        return f"group:{group_id}"
    return f"image:{row[f'{side}_image'].replace('\\', '/')}"


class DSU:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def main() -> None:
    candidates = read_csv(CANDIDATES)
    review_rows = read_csv(REVIEW)
    old_consolidation = read_csv(CONSOLIDATION) if CONSOLIDATION.exists() else []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if CONSOLIDATION.exists():
        shutil.copy2(CONSOLIDATION, REPORT_DIR / f"cross_group_consolidation_map.backup_{timestamp}.csv")
    if FAMILIES.exists():
        shutil.copy2(FAMILIES, REPORT_DIR / f"confirmed_source_photo_families.backup_{timestamp}.csv")
    if CONFLICTS.exists():
        shutil.copy2(CONFLICTS, REPORT_DIR / f"cross_group_conflict_report.backup_{timestamp}.csv")

    dsu = DSU()
    for row in review_rows:
        dsu.find(source_unit(row, "left"))
        dsu.find(source_unit(row, "right"))
        if row["decision"] == "same_photo_family":
            dsu.union(source_unit(row, "left"), source_unit(row, "right"))

    family_members: dict[str, set[str]] = defaultdict(set)
    for unit in list(dsu.parent):
        family_members[dsu.find(unit)].add(unit)
    family_rows = [
        {
            "family_root": root,
            "member_count": str(len(members)),
            "member_source_units": " | ".join(sorted(members)),
        }
        for root, members in sorted(family_members.items())
        if len(members) > 1
    ]
    write_csv(FAMILIES, family_rows, ["family_root", "member_count", "member_source_units"])

    reviewed = {pair_key(row) for row in review_rows}
    old_consolidated = {pair_key(row) for row in old_consolidation}
    consolidation_rows = []
    for index, row in enumerate(candidates, 1):
        if pair_key(row) in reviewed:
            continue
        left_unit = source_unit(row, "left")
        right_unit = source_unit(row, "right")
        if dsu.find(left_unit) != dsu.find(right_unit):
            continue
        consolidation_rows.append({
            "candidate_index_1based": str(index),
            "left_image": row["left_image"],
            "right_image": row["right_image"],
            "left_group_id": row["left_group_id"],
            "right_group_id": row["right_group_id"],
            "left_split": row["left_split"],
            "right_split": row["right_split"],
            "hash_distance": row["hash_distance"],
            "source_unit_a": left_unit,
            "source_unit_b": right_unit,
            "family_root": dsu.find(left_unit),
            "consolidation_reason": "endpoints already connected by confirmed same_photo_family review rows",
        })
    write_csv(CONSOLIDATION, consolidation_rows, [
        "candidate_index_1based",
        "left_image",
        "right_image",
        "left_group_id",
        "right_group_id",
        "left_split",
        "right_split",
        "hash_distance",
        "source_unit_a",
        "source_unit_b",
        "family_root",
        "consolidation_reason",
    ])

    conflicts = []
    for row in review_rows:
        if row["decision"] not in {"different_photos", "uncertain", "different_photos_unresolved_relationship_retained"}:
            continue
        left_unit = source_unit(row, "left")
        right_unit = source_unit(row, "right")
        if dsu.find(left_unit) == dsu.find(right_unit):
            conflicts.append({
                "review_id": row["review_id"],
                "decision": row["decision"],
                "left_image": row["left_image"],
                "right_image": row["right_image"],
                "left_unit": left_unit,
                "right_unit": right_unit,
                "family_root": dsu.find(left_unit),
                "conflict_reason": "non-same decision exists between endpoints connected by same_photo_family graph",
            })
    write_csv(CONFLICTS, conflicts, [
        "review_id",
        "decision",
        "left_image",
        "right_image",
        "left_unit",
        "right_unit",
        "family_root",
        "conflict_reason",
    ])
    print({
        "families": len(family_rows),
        "consolidated": len(consolidation_rows),
        "newly_consolidated_delta_vs_previous": len({pair_key(row) for row in consolidation_rows} - old_consolidated),
        "conflicts": len(conflicts),
    })


if __name__ == "__main__":
    main()
