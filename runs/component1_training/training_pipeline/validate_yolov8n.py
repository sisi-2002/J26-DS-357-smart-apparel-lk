"""Phase 6.2 — YOLOv8n validation and test evaluation.

Usage:
    python validate_yolov8n.py --checkpoint runs/.../best.pt --split valid
    python validate_yolov8n.py --checkpoint runs/.../best.pt --split test --authorize-test
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
DATASET_ROOT = REPO_ROOT / "data-preprocessing" / "master_fabric_dataset" / "processed" / "chenab_detection_training_v1"
RUNS_DIR = REPO_ROOT / "runs" / "component1_training"

CLASS_NAMES = ["baekra", "color issues", "contamination", "cut",
               "gray stitch", "selvet", "stain"]


def create_derived_data_yaml(output_dir: Path) -> Path:
    """Create a run-specific data.yaml with absolute dataset path."""
    dataset_path = str(DATASET_ROOT.resolve()).replace("\\", "/")
    lines = [
        "# Derived dataset config for validation",
        f"# Source: {DATASET_ROOT / 'data.yaml'}",
        "",
        f"path: {dataset_path}",
        "train: train/images",
        "val: valid/images",
        "test: test/images",
        "",
        f"nc: {len(CLASS_NAMES)}",
        f"names: {CLASS_NAMES}",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    derived = output_dir / "data.yaml"
    derived.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return derived


def validate(checkpoint: Path, split: str, output_dir: Path, imgsz: int = 640,
             batch: int = 4):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: Ultralytics not installed.")
        sys.exit(1)

    if not checkpoint.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint}")
        sys.exit(1)

    if not DATASET_ROOT.is_dir():
        print(f"ERROR: Dataset not found: {DATASET_ROOT}")
        sys.exit(1)

    derived_yaml = create_derived_data_yaml(output_dir)

    print(f"Validating: {checkpoint}")
    print(f"Split: {split}")
    print(f"Image size: {imgsz}")
    print(f"Batch: {batch}")
    print(f"Dataset root: {DATASET_ROOT}")
    print(f"Derived data.yaml: {derived_yaml}")

    model = YOLO(str(checkpoint))
    start = time.time()
    results = model.val(
        data=str(derived_yaml),
        split=split if split != "valid" else "val",
        imgsz=imgsz,
        batch=batch,
        device=0,
        workers=0,
        plots=True,
        save_json=True,
        project=str(output_dir.parent),
        name=output_dir.name,
        exist_ok=True,
    )
    elapsed = time.time() - start

    metrics = {
        "checkpoint": str(checkpoint),
        "split": split,
        "imgsz": imgsz,
        "batch": batch,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(elapsed, 1),
        "mAP50": round(float(results.box.map50), 4) if hasattr(results.box, "map50") else None,
        "mAP50_95": round(float(results.box.map), 4) if hasattr(results.box, "map") else None,
        "precision": round(float(results.box.mp), 4) if hasattr(results.box, "mp") else None,
        "recall": round(float(results.box.mr), 4) if hasattr(results.box, "mr") else None,
    }

    if hasattr(results.box, "ap50"):
        per_class = {}
        ap50 = results.box.ap50
        for i, name in enumerate(CLASS_NAMES):
            if i < len(ap50):
                per_class[name] = {"ap50": round(float(ap50[i]), 4)}
        metrics["per_class"] = per_class

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"validation_{split}_metrics.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'=' * 50}")
    print(f"Validation Results — {split}")
    print(f"{'=' * 50}")
    print(f"mAP50:    {metrics.get('mAP50', 'N/A')}")
    print(f"mAP50-95: {metrics.get('mAP50_95', 'N/A')}")
    print(f"Precision:{metrics.get('precision', 'N/A')}")
    print(f"Recall:   {metrics.get('recall', 'N/A')}")
    if "per_class" in metrics:
        print(f"\nPer-class AP50:")
        for name, vals in metrics["per_class"].items():
            print(f"  {name}: {vals['ap50']}")
    print(f"\nSaved: {report_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="YOLOv8n Validation")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", choices=["valid", "test"], default="valid")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Validation image size (match training config)")
    parser.add_argument("--batch", type=int, default=4,
                        help="Validation batch size")
    parser.add_argument("--authorize-test", action="store_true",
                        help="Explicitly authorize locked test evaluation")
    parser.add_argument("--output", type=Path, default=None)

    args = parser.parse_args()

    if args.split == "test" and not args.authorize_test:
        print("ERROR: Test evaluation requires --authorize-test flag.")
        print("The test split is locked until model selection is frozen.")
        print("Use --split valid for development evaluation.")
        sys.exit(1)

    if args.split == "test":
        print("=" * 60)
        print("WARNING: LOCKED TEST EVALUATION")
        print("This should only be done ONCE after final model selection.")
        print("=" * 60)

    if not args.checkpoint.is_absolute():
        args.checkpoint = Path.cwd() / args.checkpoint

    if args.output:
        output_dir = args.output
    else:
        exp_dir = args.checkpoint.parent.parent
        output_dir = exp_dir / f"eval_{args.split}"

    validate(args.checkpoint, args.split, output_dir, imgsz=args.imgsz,
             batch=args.batch)


if __name__ == "__main__":
    main()
