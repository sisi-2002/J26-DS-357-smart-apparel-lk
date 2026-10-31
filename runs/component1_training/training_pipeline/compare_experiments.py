"""Phase 6.2 — Compare completed training experiments.

Usage:
    python compare_experiments.py
    python compare_experiments.py --runs-dir runs/component1_training
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
DEFAULT_RUNS = REPO_ROOT / "runs" / "component1_training"
REPORT_DIR = SCRIPT_DIR / "reports"


def find_experiments(runs_dir: Path) -> list[dict]:
    experiments = []
    if not runs_dir.is_dir():
        return experiments

    for entry in sorted(runs_dir.iterdir()):
        if not entry.is_dir():
            continue
        meta_path = entry / "experiment_metadata.json"
        results_csv = entry / "results.csv"
        valid_metrics = entry / "eval_valid" / "validation_valid_metrics.json"

        exp: dict = {"dir": str(entry), "name": entry.name}

        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            exp["experiment_id"] = meta.get("experiment_id", entry.name)
            exp["config"] = meta.get("config", {})
            exp["duration_seconds"] = meta.get("training_duration_seconds")
            exp["peak_gpu_mb"] = meta.get("peak_gpu_memory_mb")
        else:
            exp["experiment_id"] = entry.name
            exp["config"] = {}

        args_yaml = entry / "args.yaml"
        if args_yaml.exists():
            try:
                from train_yolov8n import _load_yaml_fallback
                exp["ultralytics_args"] = _load_yaml_fallback(args_yaml)
            except Exception:
                pass

        if valid_metrics.exists():
            with open(valid_metrics, encoding="utf-8") as f:
                vm = json.load(f)
            exp["mAP50"] = vm.get("mAP50")
            exp["mAP50_95"] = vm.get("mAP50_95")
            exp["precision"] = vm.get("precision")
            exp["recall"] = vm.get("recall")
            exp["per_class"] = vm.get("per_class", {})
        elif results_csv.exists():
            exp.update(_parse_results_csv(results_csv))
        else:
            exp["mAP50"] = None
            exp["mAP50_95"] = None

        best_pt = entry / "weights" / "best.pt"
        if best_pt.exists():
            exp["best_pt_size_mb"] = round(best_pt.stat().st_size / 1024**2, 1)
            exp["complete"] = True
        else:
            exp["best_pt_size_mb"] = None
            exp["complete"] = False

        experiments.append(exp)

    return experiments


def _parse_results_csv(path: Path) -> dict:
    """Parse Ultralytics results.csv for final-epoch metrics."""
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}

    last = rows[-1]
    best_map = 0.0
    best_row = last
    for r in rows:
        for key in r:
            if "mAP50-95" in key or "metrics/mAP50-95(B)" in key:
                try:
                    v = float(r[key].strip())
                    if v > best_map:
                        best_map = v
                        best_row = r
                except (ValueError, KeyError):
                    pass

    result: dict = {}
    for key in best_row:
        k = key.strip()
        try:
            val = float(best_row[key].strip())
        except (ValueError, AttributeError):
            continue
        if "mAP50-95" in k:
            result["mAP50_95"] = round(val, 4)
        elif "mAP50" in k and "95" not in k:
            result["mAP50"] = round(val, 4)
        elif "precision" in k.lower():
            result["precision"] = round(val, 4)
        elif "recall" in k.lower():
            result["recall"] = round(val, 4)

    return result


def generate_comparison(experiments: list[dict], output_dir: Path):
    if not experiments:
        print("No experiments found.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "experiment_comparison.csv"
    md_path = output_dir / "experiment_comparison.md"

    fields = ["experiment_id", "imgsz", "batch", "epochs", "optimizer", "lr0",
              "mAP50", "mAP50_95", "precision", "recall",
              "best_pt_size_mb", "duration_seconds", "peak_gpu_mb"]

    rows = []
    for exp in experiments:
        row = {
            "experiment_id": exp["experiment_id"],
            "imgsz": exp.get("config", {}).get("imgsz", ""),
            "batch": exp.get("config", {}).get("batch", ""),
            "epochs": exp.get("config", {}).get("epochs", ""),
            "optimizer": exp.get("config", {}).get("optimizer", ""),
            "lr0": exp.get("config", {}).get("lr0", ""),
            "mAP50": exp.get("mAP50", ""),
            "mAP50_95": exp.get("mAP50_95", ""),
            "precision": exp.get("precision", ""),
            "recall": exp.get("recall", ""),
            "best_pt_size_mb": exp.get("best_pt_size_mb", ""),
            "duration_seconds": exp.get("duration_seconds", ""),
            "peak_gpu_mb": exp.get("peak_gpu_mb", ""),
        }
        rows.append(row)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Experiment Comparison Report",
        f"\nGenerated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"\nExperiments found: {len(experiments)}",
        "\n## Summary Table\n",
        "| Experiment | ImgSz | Batch | Epochs | Optimizer | LR | mAP50 | mAP50-95 | P | R | Size (MB) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        line = (
            f"| {r['experiment_id']} | {r['imgsz']} | {r['batch']} | {r['epochs']} "
            f"| {r['optimizer']} | {r['lr0']} | {r['mAP50']} | {r['mAP50_95']} "
            f"| {r['precision']} | {r['recall']} | {r['best_pt_size_mb']} |"
        )
        lines.append(line)

    ranked = sorted(
        [e for e in experiments if e.get("mAP50_95") is not None],
        key=lambda x: x["mAP50_95"],
        reverse=True,
    )
    if ranked:
        lines.append(f"\n## Ranking by Validation mAP50-95\n")
        for i, exp in enumerate(ranked, 1):
            lines.append(f"{i}. **{exp['experiment_id']}** — mAP50-95: {exp['mAP50_95']}")

        lines.append(f"\n**Best candidate:** {ranked[0]['experiment_id']} "
                      f"(mAP50-95 = {ranked[0]['mAP50_95']})")

    for exp in experiments:
        if exp.get("per_class"):
            lines.append(f"\n### Per-class AP50 — {exp['experiment_id']}\n")
            lines.append("| Class | AP50 |")
            lines.append("|---|---|")
            for cls, vals in exp["per_class"].items():
                lines.append(f"| {cls} | {vals.get('ap50', 'N/A')} |")

    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Comparison CSV: {csv_path}")
    print(f"Comparison report: {md_path}")

    for line in lines:
        print(line)


def main():
    parser = argparse.ArgumentParser(description="Compare YOLOv8n experiments")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS)
    args = parser.parse_args()

    experiments = find_experiments(args.runs_dir)
    comparison_dir = args.runs_dir / "comparison"
    generate_comparison(experiments, comparison_dir)


if __name__ == "__main__":
    main()
