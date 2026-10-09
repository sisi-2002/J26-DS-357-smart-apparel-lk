"""Phase 6.2 — Select the best YOLOv8n model from completed experiments.

Usage:
    python select_best_model.py
    python select_best_model.py --runs-dir runs/component1_training
"""

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
DEFAULT_RUNS = REPO_ROOT / "runs" / "component1_training"
REPORT_DIR = SCRIPT_DIR / "reports"


def main():
    parser = argparse.ArgumentParser(description="Select best YOLOv8n model")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS)
    args = parser.parse_args()

    from compare_experiments import find_experiments
    experiments = find_experiments(args.runs_dir)

    scored = [e for e in experiments
              if e.get("mAP50_95") is not None and e.get("complete", False)]
    if not scored:
        print("No experiments with validation metrics found.")
        print("Run training experiments and validation first.")
        sys.exit(1)

    ranked = sorted(scored, key=lambda x: x["mAP50_95"], reverse=True)
    best = ranked[0]

    selection = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "criterion": "validation_mAP50_95",
        "selected_experiment": best["experiment_id"],
        "selected_dir": best["dir"],
        "mAP50_95": best["mAP50_95"],
        "mAP50": best.get("mAP50"),
        "precision": best.get("precision"),
        "recall": best.get("recall"),
        "model_size_mb": best.get("best_pt_size_mb"),
        "all_candidates": [
            {
                "experiment_id": e["experiment_id"],
                "mAP50_95": e.get("mAP50_95"),
                "mAP50": e.get("mAP50"),
            }
            for e in ranked
        ],
    }

    if best.get("per_class"):
        selection["per_class"] = best["per_class"]
        weak = []
        for cls, vals in best["per_class"].items():
            if vals.get("ap50", 1.0) < 0.5:
                weak.append(cls)
        if weak:
            selection["weak_classes_warning"] = weak

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "model_selection_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(selection, f, indent=2)

    md_lines = [
        "# Model Selection Report",
        f"\nGenerated: {selection['timestamp']}",
        f"\n## Selected Model",
        f"\n- **Experiment:** {best['experiment_id']}",
        f"- **Validation mAP50-95:** {best['mAP50_95']}",
        f"- **Validation mAP50:** {best.get('mAP50', 'N/A')}",
        f"- **Precision:** {best.get('precision', 'N/A')}",
        f"- **Recall:** {best.get('recall', 'N/A')}",
        f"- **Model size:** {best.get('best_pt_size_mb', 'N/A')} MB",
        f"- **Checkpoint:** {best['dir']}/weights/best.pt",
        f"\n## Selection Criterion",
        f"\nPrimary: highest validation mAP50-95",
        f"\n## All Candidates\n",
        "| Rank | Experiment | mAP50-95 | mAP50 |",
        "|---|---|---|---|",
    ]
    for i, e in enumerate(ranked, 1):
        marker = " **←**" if i == 1 else ""
        md_lines.append(
            f"| {i} | {e['experiment_id']} | {e.get('mAP50_95', 'N/A')} | {e.get('mAP50', 'N/A')} |{marker}"
        )

    if selection.get("weak_classes_warning"):
        md_lines.append(f"\n## Weak Classes Warning")
        md_lines.append(f"\nThe following classes have AP50 < 0.5:")
        for cls in selection["weak_classes_warning"]:
            md_lines.append(f"- {cls}")

    md_lines.append(f"\n## Next Steps")
    md_lines.append(f"\n1. Review per-class metrics for the selected model.")
    md_lines.append(f"2. Freeze this selection before test evaluation.")
    md_lines.append(f"3. Run locked test evaluation with --authorize-test.")

    md_path = REPORT_DIR / "model_selection_report.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"\n{'=' * 50}")
    print(f"BEST MODEL: {best['experiment_id']}")
    print(f"  mAP50-95: {best['mAP50_95']}")
    print(f"  mAP50:    {best.get('mAP50', 'N/A')}")
    print(f"  Precision:{best.get('precision', 'N/A')}")
    print(f"  Recall:   {best.get('recall', 'N/A')}")
    print(f"{'=' * 50}")
    print(f"\nReports saved to: {REPORT_DIR}")


if __name__ == "__main__":
    main()
