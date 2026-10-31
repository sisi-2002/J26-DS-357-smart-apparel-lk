"""Step 7: Train YOLOv8n baseline on fabricDD_processed_20261011."""
import json, os, sys, platform
from datetime import datetime
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────
BASE = Path(r"D:\Fabric defect detection\J26-DS-357-smart-apparel-lk\data-preprocessing")
DATASET = BASE / "fabricDD_processed_20261011"
DATA_YAML = DATASET / "data_training.yaml"
REPORTS = BASE / "reports"

SEED = 42
MODEL = "yolov8n.pt"
IMGSZ = 640
EPOCHS = 50
BATCH = 8  # conservative for 4GB VRAM RTX 3050
WORKERS = 4

SMOKE_TEST = "--smoke" in sys.argv
if SMOKE_TEST:
    EPOCHS = 2
    PROJECT_NAME = "baseline_smoke"
else:
    PROJECT_NAME = "baseline_v1"

RUN_DIR = BASE / "training_runs"
PROJECT = RUN_DIR / PROJECT_NAME

# ── Environment info ─────────────────────────────────────────────────────────
import torch
import ultralytics

env_info = {
    "python_version": sys.version,
    "python_executable": sys.executable,
    "platform": platform.platform(),
    "torch_version": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_mem / 1024**3, 1) if torch.cuda.is_available() else None,
    "ultralytics_version": ultralytics.__version__,
    "timestamp": datetime.now().isoformat(),
}

device = "0" if torch.cuda.is_available() else "cpu"
print(f"Device: {device} ({env_info.get('gpu_name', 'CPU')})")
print(f"PyTorch: {torch.__version__}, CUDA: {env_info.get('cuda_version', 'N/A')}")
print(f"Ultralytics: {ultralytics.__version__}")
print(f"Mode: {'SMOKE TEST' if SMOKE_TEST else 'FULL TRAINING'}")

# ── Train ────────────────────────────────────────────────────────────────────
from ultralytics import YOLO

model = YOLO(MODEL)

train_args = dict(
    data=str(DATA_YAML),
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    seed=SEED,
    device=device,
    workers=WORKERS,
    project=str(PROJECT),
    name="train",
    exist_ok=False,
    patience=10,
    save=True,
    save_period=-1,
    val=True,
    plots=True,
    verbose=True,
)

# Save config before training
config = {
    "model": MODEL,
    "data_yaml": str(DATA_YAML),
    "dataset": str(DATASET),
    **{k: str(v) if isinstance(v, Path) else v for k, v in train_args.items()},
    "environment": env_info,
}

os.makedirs(PROJECT, exist_ok=True)
config_path = PROJECT / "training_config.json"
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)
print(f"\nConfig saved: {config_path}")
print(f"Starting training: {EPOCHS} epochs, batch={BATCH}, imgsz={IMGSZ}")

try:
    results = model.train(**train_args)
except RuntimeError as e:
    if "out of memory" in str(e).lower() or "CUDA" in str(e):
        print(f"\nGPU OOM with batch={BATCH}. Reducing to batch={BATCH//2}.")
        torch.cuda.empty_cache()
        BATCH = BATCH // 2
        train_args["batch"] = BATCH
        train_args["name"] = "train"
        config["batch"] = BATCH
        config["note"] = f"Reduced batch from {BATCH*2} to {BATCH} due to OOM"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        results = model.train(**train_args)
    else:
        raise

# ── Post-training summary ───────────────────────────────────────────────────
run_dir = Path(results.save_dir) if hasattr(results, 'save_dir') else PROJECT / "train"
print(f"\nRun directory: {run_dir}")

# Check for best.pt and last.pt
best_pt = run_dir / "weights" / "best.pt"
last_pt = run_dir / "weights" / "last.pt"
print(f"best.pt: {'EXISTS' if best_pt.exists() else 'MISSING'}")
print(f"last.pt: {'EXISTS' if last_pt.exists() else 'MISSING'}")

# Read results
results_csv = run_dir / "results.csv"
if results_csv.exists():
    import csv
    with open(results_csv) as f:
        rows = list(csv.DictReader(f))
    if rows:
        last = rows[-1]
        # Clean up keys (ultralytics adds spaces)
        last = {k.strip(): v.strip() for k, v in last.items()}
        print(f"\nFinal epoch results:")
        for key in ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]:
            if key in last:
                print(f"  {key}: {last[key]}")

# Save summary
summary = {
    "status": "completed",
    "run_dir": str(run_dir),
    "epochs_completed": EPOCHS,
    "best_pt": str(best_pt),
    "last_pt": str(last_pt),
    "config": config,
    "resume_command": f'python 10_train_baseline.py  # or: model.train(resume=True) from {last_pt}',
}
summary_path = run_dir / "training_summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nTraining summary saved: {summary_path}")
print("DONE")
