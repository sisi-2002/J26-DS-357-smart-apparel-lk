"""Phase 6.2 — Manual YOLOv8n training entry point.

Usage:
    python train_yolov8n.py --config configs/e01_baseline.yaml --dry-run
    python train_yolov8n.py --config configs/e01_baseline.yaml --execute
    python train_yolov8n.py --config configs/e01_baseline.yaml --resume runs/component1_training/E01_baseline
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
DATASET_ROOT = REPO_ROOT / "data-preprocessing" / "master_fabric_dataset" / "processed" / "chenab_detection_training_v1"
RUNS_DIR = REPO_ROOT / "runs" / "component1_training"
AUG_POLICY = DATASET_ROOT / "augmentation" / "augmentation_policy.json"
AUG_YAML = DATASET_ROOT / "augmentation" / "yolo_training_augmentation_config.yaml"

BLOCKED_CHECKPOINTS = [
    "backend/app/model/best.pt",
    "backend/app/model/last.pt",
]

REQUIRED_KEYS = ["experiment_id", "model", "imgsz", "epochs", "batch", "optimizer",
                 "lr0", "patience", "seed"]

CLASS_NAMES = ["baekra", "color issues", "contamination", "cut",
               "gray stitch", "selvet", "stain"]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_hash(path: Path) -> str:
    if path.exists():
        return _sha256(path)
    return "FILE_NOT_FOUND"


def load_config(config_path: Path) -> dict:
    try:
        import yaml
        with open(config_path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    except ImportError:
        return _load_yaml_fallback(config_path)


def _load_yaml_fallback(path: Path) -> dict:
    """Minimal YAML parser for flat key-value configs (no nested structures)."""
    cfg = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                continue
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip()
            if val.startswith("#"):
                val = ""
            if "#" in val:
                val = val[:val.index("#")].strip()
            if val.lower() == "true":
                cfg[key] = True
            elif val.lower() == "false":
                cfg[key] = False
            elif val.lower() == "null" or val.lower() == "none" or val == "":
                cfg[key] = None
            else:
                try:
                    cfg[key] = int(val)
                except ValueError:
                    try:
                        cfg[key] = float(val)
                    except ValueError:
                        cfg[key] = val
    return cfg


def load_augmentation_params() -> dict:
    """Load augmentation parameters from Phase 5 policy."""
    with open(AUG_POLICY, encoding="utf-8") as f:
        policy = json.load(f)

    params = {}
    mapping = {
        "fliplr": ("fliplr", "probability"),
        "flipud": ("flipud", "probability"),
        "degrees": ("degrees", "value"),
        "translate": ("translate", "value"),
        "scale": ("scale", "value"),
        "hsv_h": ("hsv_h", "value"),
        "hsv_s": ("hsv_s", "value"),
        "hsv_v": ("hsv_v", "value"),
        "mosaic": ("mosaic", "probability"),
        "shear": ("shear", "value"),
        "perspective": ("perspective", "value"),
        "mixup": ("mixup", "probability"),
        "copy_paste": ("copy_paste", "probability"),
    }
    for yolo_key, (policy_key, field) in mapping.items():
        t = policy["transforms"].get(policy_key, {})
        params[yolo_key] = t.get(field, 0.0)
    return params


def validate_config(cfg: dict) -> list[str]:
    errors = []
    for k in REQUIRED_KEYS:
        if k not in cfg:
            errors.append(f"Missing required key: {k}")

    if cfg.get("model", "") != "yolov8n.pt":
        model_name = cfg.get("model", "")
        if model_name not in ("yolov8n.pt",):
            errors.append(
                f"Model must be 'yolov8n.pt' (official pretrained). Got: {model_name}"
            )

    if cfg.get("imgsz", 640) > 960:
        errors.append(f"Image size {cfg.get('imgsz')} too large for 4 GB VRAM.")

    if cfg.get("batch", 4) > 16:
        errors.append(f"Batch size {cfg.get('batch')} is dangerously large for 4 GB VRAM.")

    if cfg.get("epochs", 150) > 500:
        errors.append(f"Epochs {cfg.get('epochs')} exceeds reasonable limit.")

    if not DATASET_ROOT.is_dir():
        errors.append(f"Dataset directory not found: {DATASET_ROOT}")

    if not (DATASET_ROOT / "data.yaml").exists():
        errors.append("data.yaml not found in dataset directory.")

    if not AUG_POLICY.exists():
        errors.append(f"Augmentation policy not found: {AUG_POLICY}")

    for blocked in BLOCKED_CHECKPOINTS:
        bp = REPO_ROOT / blocked
        if bp.exists() and cfg.get("model", "") in (blocked, str(bp), bp.name):
            errors.append(
                f"BLOCKED: Cannot use historical checkpoint {blocked}. "
                "Use official yolov8n.pt only."
            )

    return errors


def validate_dataset() -> list[str]:
    errors = []
    for split in ("train", "valid"):
        img_dir = DATASET_ROOT / split / "images"
        lbl_dir = DATASET_ROOT / split / "labels"
        if not img_dir.is_dir():
            errors.append(f"Missing {split}/images directory")
        if not lbl_dir.is_dir():
            errors.append(f"Missing {split}/labels directory")

    train_imgs = DATASET_ROOT / "train" / "images"
    if train_imgs.is_dir():
        count = len([f for f in os.listdir(train_imgs) if f.lower().endswith((".jpg", ".png", ".jpeg"))])
        if count != 1900:
            errors.append(f"Expected 1900 training images, found {count}")

    valid_imgs = DATASET_ROOT / "valid" / "images"
    if valid_imgs.is_dir():
        count = len([f for f in os.listdir(valid_imgs) if f.lower().endswith((".jpg", ".png", ".jpeg"))])
        if count != 377:
            errors.append(f"Expected 377 validation images, found {count}")

    return errors


def create_derived_data_yaml(output_dir: Path) -> Path:
    """Create a run-specific data.yaml with absolute dataset path.

    The frozen data.yaml has `path: .` which Ultralytics resolves relative to
    the YAML file location or CWD. This function writes a derived copy with
    the absolute dataset root so training works from any working directory.
    The frozen original is never modified.
    """
    dataset_path = str(DATASET_ROOT.resolve()).replace("\\", "/")
    lines = [
        "# Derived dataset config — auto-generated for this experiment",
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


def get_output_dir(experiment_id: str) -> Path:
    out = RUNS_DIR / experiment_id
    if out.exists():
        i = 2
        while (RUNS_DIR / f"{experiment_id}_v{i}").exists():
            i += 1
        out = RUNS_DIR / f"{experiment_id}_v{i}"
    return out


def collect_metadata(cfg: dict, output_dir: Path) -> dict:
    meta = {
        "experiment_id": cfg.get("experiment_id"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "system": {
            "os": f"{platform.system()} {platform.release()}",
            "python": platform.python_version(),
            "interpreter": sys.executable,
        },
        "config": cfg,
        "dataset": {
            "path": str(DATASET_ROOT),
            "version": "chenab_detection_training_v1",
            "data_yaml_hash": _file_hash(DATASET_ROOT / "data.yaml"),
        },
        "augmentation": {
            "policy_hash": _file_hash(AUG_POLICY),
            "yaml_hash": _file_hash(AUG_YAML),
        },
        "output_dir": str(output_dir),
    }

    for pkg in ["torch", "ultralytics", "numpy", "opencv-python-headless", "pyyaml"]:
        try:
            meta["system"][f"pkg_{pkg}"] = importlib.metadata.version(pkg)
        except Exception:
            meta["system"][f"pkg_{pkg}"] = "NOT_INSTALLED"

    try:
        import torch
        if torch.cuda.is_available():
            meta["system"]["gpu"] = torch.cuda.get_device_name(0)
            meta["system"]["vram_mb"] = round(
                torch.cuda.get_device_properties(0).total_mem / 1024**2
            )
    except Exception:
        pass

    return meta


def build_train_args(cfg: dict, aug_params: dict, data_yaml: Path, output_dir: Path) -> dict:
    """Build the kwargs dict for YOLO model.train()."""
    args = {
        "data": str(data_yaml),
        "imgsz": cfg.get("imgsz", 640),
        "epochs": cfg.get("epochs", 150),
        "batch": cfg.get("batch", 4),
        "device": 0,
        "optimizer": cfg.get("optimizer", "AdamW"),
        "lr0": cfg.get("lr0", 0.001),
        "patience": cfg.get("patience", 35),
        "seed": cfg.get("seed", 42),
        "deterministic": cfg.get("deterministic", True),
        "amp": cfg.get("amp", True),
        "cos_lr": cfg.get("cos_lr", True),
        "warmup_epochs": cfg.get("warmup_epochs", 3.0),
        "workers": cfg.get("workers", 0),
        "cache": cfg.get("cache", False),
        "project": str(RUNS_DIR),
        "name": output_dir.name,
        "exist_ok": True,
        "save": True,
        "save_period": cfg.get("save_period", -1),
        "plots": True,
        "val": True,
        "verbose": True,
    }

    for key, val in aug_params.items():
        args[key] = val

    if cfg.get("lrf") is not None:
        args["lrf"] = cfg["lrf"]
    if cfg.get("weight_decay") is not None:
        args["weight_decay"] = cfg["weight_decay"]
    if cfg.get("momentum") is not None:
        args["momentum"] = cfg["momentum"]
    if cfg.get("close_mosaic") is not None:
        args["close_mosaic"] = cfg["close_mosaic"]

    return args


def dry_run(cfg: dict):
    print("\n" + "=" * 60)
    print("DRY RUN — No training will occur")
    print("=" * 60)

    errors = validate_config(cfg)
    if errors:
        print("\nConfiguration errors:")
        for e in errors:
            print(f"  ERROR: {e}")
        return False

    ds_errors = validate_dataset()
    if ds_errors:
        print("\nDataset errors:")
        for e in ds_errors:
            print(f"  ERROR: {e}")
        return False

    aug_params = load_augmentation_params()
    output_dir = get_output_dir(cfg["experiment_id"])

    import tempfile
    dry_dir = Path(tempfile.mkdtemp(prefix="yolo_dryrun_"))
    derived_yaml = create_derived_data_yaml(dry_dir)
    train_args = build_train_args(cfg, aug_params, derived_yaml, output_dir)

    print(f"\nExperiment ID: {cfg['experiment_id']}")
    print(f"Model: {cfg.get('model', 'yolov8n.pt')}")
    print(f"Dataset root: {DATASET_ROOT}")
    print(f"Derived data.yaml: {derived_yaml}")
    print(f"Output: {output_dir}")
    print(f"\nTraining arguments:")
    for k, v in sorted(train_args.items()):
        print(f"  {k}: {v}")

    print(f"\nAugmentation (from Phase 5 policy):")
    for k, v in sorted(aug_params.items()):
        print(f"  {k}: {v}")

    print(f"\nDataset path resolution check:")
    ds_root = DATASET_ROOT.resolve()
    for split_name, split_rel in [("train", "train/images"), ("val", "valid/images")]:
        resolved = ds_root / split_rel.replace("/", os.sep)
        exists = resolved.is_dir()
        count = len(os.listdir(resolved)) if exists else 0
        status = f"[OK] {count} files" if exists else "[FAIL] NOT FOUND"
        print(f"  {split_name}: {resolved} -> {status}")
        if not exists:
            print(f"\n  ERROR: Ultralytics would fail to find {split_name} images.")
            return False

    meta = collect_metadata(cfg, output_dir)
    print(f"\nMetadata preview:")
    print(f"  Dataset hash: {meta['dataset']['data_yaml_hash'][:16]}...")
    print(f"  Aug policy hash: {meta['augmentation']['policy_hash'][:16]}...")

    print("\n[OK] Configuration valid")
    print("[OK] Dataset accessible")
    print("[OK] Augmentation policy loaded")
    print("[OK] Dataset paths resolve correctly")
    print(f"[OK] Output directory: {output_dir}")
    print("\nTo start training, run with --execute")
    return True


def execute_training(cfg: dict):
    print("\n" + "=" * 60)
    print("EXECUTE — Starting YOLOv8n training")
    print("=" * 60)

    errors = validate_config(cfg)
    if errors:
        print("\nConfiguration errors — ABORTING:")
        for e in errors:
            print(f"  ERROR: {e}")
        sys.exit(1)

    ds_errors = validate_dataset()
    if ds_errors:
        print("\nDataset errors — ABORTING:")
        for e in ds_errors:
            print(f"  ERROR: {e}")
        sys.exit(1)

    try:
        import torch
        if not torch.cuda.is_available():
            print("\nERROR: CUDA not available. GPU training requires CUDA.")
            print("Check your PyTorch installation and GPU drivers.")
            sys.exit(1)
    except ImportError:
        print("\nERROR: PyTorch not installed.")
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("\nERROR: Ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    aug_params = load_augmentation_params()
    output_dir = get_output_dir(cfg["experiment_id"])
    derived_yaml = create_derived_data_yaml(output_dir)
    train_args = build_train_args(cfg, aug_params, derived_yaml, output_dir)
    meta = collect_metadata(cfg, output_dir)
    meta["derived_data_yaml"] = str(derived_yaml)

    print(f"\nExperiment: {cfg['experiment_id']}")
    print(f"Output: {output_dir}")
    print(f"Derived data.yaml: {derived_yaml}")
    print(f"Device: GPU 0 ({torch.cuda.get_device_name(0)})")
    print(f"Model: {cfg.get('model', 'yolov8n.pt')}")
    print(f"Image size: {cfg.get('imgsz', 640)}")
    print(f"Batch: {cfg.get('batch', 4)}")
    print(f"Epochs: {cfg.get('epochs', 150)}")
    print(f"Optimizer: {cfg.get('optimizer', 'AdamW')}")
    print(f"LR: {cfg.get('lr0', 0.001)}")

    start_time = time.time()

    try:
        model = YOLO(cfg.get("model", "yolov8n.pt"))
        results = model.train(**train_args)
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "CUDA" in str(e):
            elapsed = time.time() - start_time
            print(f"\n{'=' * 60}")
            print("CUDA OUT OF MEMORY")
            print(f"{'=' * 60}")
            print(f"Failed after {elapsed:.0f}s")
            print(f"\nCurrent settings:")
            print(f"  Batch size: {cfg.get('batch', 4)}")
            print(f"  Image size: {cfg.get('imgsz', 640)}")
            print(f"\nRecovery options:")
            print(f"  1. Reduce batch size (try {max(1, cfg.get('batch', 4) // 2)})")
            print(f"  2. Reduce image size (try {cfg.get('imgsz', 640) - 64})")
            print(f"  3. Close other GPU applications")
            print(f"  4. Ensure AMP is enabled (amp: true)")
            print(f"\nDo NOT silently change settings — update the config and re-run.")
            sys.exit(1)
        raise

    elapsed = time.time() - start_time
    meta["training_duration_seconds"] = round(elapsed)
    meta["training_duration_human"] = f"{elapsed/3600:.1f} hours"

    try:
        meta["peak_gpu_memory_mb"] = round(torch.cuda.max_memory_allocated(0) / 1024**2)
    except Exception:
        pass

    with open(output_dir / "experiment_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print("TRAINING COMPLETE")
    print(f"{'=' * 60}")
    print(f"Duration: {meta['training_duration_human']}")
    print(f"Results: {output_dir}")
    print(f"Best checkpoint: {output_dir / 'weights' / 'best.pt'}")
    print(f"Last checkpoint: {output_dir / 'weights' / 'last.pt'}")


def resume_training(resume_dir: Path, cfg: dict):
    print("\n" + "=" * 60)
    print("RESUME — Continuing interrupted training")
    print("=" * 60)

    last_pt = resume_dir / "weights" / "last.pt"
    if not last_pt.exists():
        print(f"\nERROR: No last.pt found at {last_pt}")
        print("Cannot resume — the checkpoint does not exist.")
        sys.exit(1)

    meta_path = resume_dir / "experiment_metadata.json"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            saved_meta = json.load(f)
        saved_id = saved_meta.get("experiment_id")
        cfg_id = cfg.get("experiment_id")
        if saved_id and cfg_id and saved_id != cfg_id:
            print(f"\nERROR: Config experiment_id '{cfg_id}' does not match")
            print(f"the saved experiment '{saved_id}' in {resume_dir}.")
            print("Use the correct config for the experiment you want to resume.")
            sys.exit(1)
        print(f"Verified experiment: {saved_id}")

    try:
        import torch
        if not torch.cuda.is_available():
            print("\nERROR: CUDA not available. GPU training requires CUDA.")
            sys.exit(1)
    except ImportError:
        print("\nERROR: PyTorch not installed.")
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("\nERROR: Ultralytics not installed.")
        sys.exit(1)

    print(f"Resuming from: {last_pt}")
    model = YOLO(str(last_pt))
    results = model.train(resume=True)

    print(f"\nRESUMED TRAINING COMPLETE")
    print(f"Results: {resume_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 6 — Manual YOLOv8n Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python train_yolov8n.py --config configs/e01_baseline.yaml --dry-run\n"
            "  python train_yolov8n.py --config configs/e01_baseline.yaml --execute\n"
            "  python train_yolov8n.py --config configs/e01_baseline.yaml --resume runs/component1_training/E01_baseline\n"
        ),
    )
    parser.add_argument("--config", required=True, type=Path, help="YAML config file")
    parser.add_argument("--dry-run", action="store_true", help="Validate without training")
    parser.add_argument("--execute", action="store_true", help="Start real training")
    parser.add_argument("--resume", type=Path, help="Resume from interrupted run directory")

    args = parser.parse_args()

    if not args.dry_run and not args.execute and not args.resume:
        print("ERROR: Specify --dry-run, --execute, or --resume <dir>")
        print("Training will NOT start without explicit authorization.")
        sys.exit(1)

    if args.execute and args.resume:
        print("ERROR: Use --execute for new training or --resume for interrupted — not both.")
        sys.exit(1)

    config_path = args.config
    if not config_path.is_absolute():
        config_path = SCRIPT_DIR / config_path
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)

    cfg = load_config(config_path)

    if args.dry_run:
        ok = dry_run(cfg)
        sys.exit(0 if ok else 1)
    elif args.resume:
        resume_dir = args.resume
        if not resume_dir.is_absolute():
            resume_dir = Path.cwd() / resume_dir
        resume_training(resume_dir, cfg)
    elif args.execute:
        execute_training(cfg)


if __name__ == "__main__":
    main()
