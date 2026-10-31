"""Phase 6.1 — GPU, CUDA and dependency environment verification.

Generates: reports/environment_report.md
"""

import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
DATASET_DIR = REPO_ROOT / "data-preprocessing" / "master_fabric_dataset" / "processed" / "chenab_detection_training_v1"
REPORT_DIR = SCRIPT_DIR / "reports"


def _run(cmd: list[str]) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return r.stdout.strip()
    except Exception as e:
        return f"ERROR: {e}"


def _pkg_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT INSTALLED"


def check_system() -> dict:
    return {
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "arch": platform.machine(),
        "python_version": platform.python_version(),
        "python_path": sys.executable,
    }


def check_nvidia() -> dict:
    info: dict = {}
    smi = _run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version",
                 "--format=csv,noheader,nounits"])
    if smi.startswith("ERROR"):
        info["nvidia_smi"] = smi
        info["gpu_name"] = "UNKNOWN"
        info["vram_total_mb"] = 0
        info["vram_free_mb"] = 0
        info["driver_version"] = "UNKNOWN"
    else:
        parts = [p.strip() for p in smi.split(",")]
        info["gpu_name"] = parts[0] if len(parts) > 0 else "UNKNOWN"
        info["vram_total_mb"] = int(parts[1]) if len(parts) > 1 else 0
        info["vram_free_mb"] = int(parts[2]) if len(parts) > 2 else 0
        info["driver_version"] = parts[3] if len(parts) > 3 else "UNKNOWN"
    return info


def check_cuda() -> dict:
    info: dict = {}
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_build_version"] = torch.version.cuda or "NONE"
        if torch.cuda.is_available():
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            info["cuda_device_count"] = torch.cuda.device_count()
            total = torch.cuda.get_device_properties(0).total_memory
            info["torch_vram_total_mb"] = round(total / 1024**2)
        else:
            info["cuda_device_name"] = "N/A"
            info["cuda_device_count"] = 0
            info["torch_vram_total_mb"] = 0
    except ImportError:
        info["torch_version"] = "NOT INSTALLED"
        info["cuda_available"] = False
        info["cuda_build_version"] = "N/A"
    return info


def check_packages() -> dict:
    names = ["torch", "torchvision", "ultralytics", "numpy", "opencv-python",
             "opencv-python-headless", "matplotlib", "pandas", "pyyaml",
             "scipy", "pillow", "albumentations", "tqdm", "psutil"]
    return {n: _pkg_version(n) for n in names}


def check_ultralytics_compat() -> dict:
    info: dict = {}
    try:
        from ultralytics import YOLO
        info["ultralytics_importable"] = True
        info["ultralytics_version"] = _pkg_version("ultralytics")
    except ImportError:
        info["ultralytics_importable"] = False
        info["ultralytics_version"] = "NOT INSTALLED"
    return info


def check_dataset() -> dict:
    info: dict = {}
    info["dataset_dir_exists"] = DATASET_DIR.is_dir()
    data_yaml = DATASET_DIR / "data.yaml"
    info["data_yaml_exists"] = data_yaml.exists()

    for split in ("train", "valid", "test"):
        img_dir = DATASET_DIR / split / "images"
        lbl_dir = DATASET_DIR / split / "labels"
        if img_dir.is_dir():
            imgs = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
            info[f"{split}_images"] = len(imgs)
        else:
            info[f"{split}_images"] = 0
        if lbl_dir.is_dir():
            lbls = [f for f in os.listdir(lbl_dir) if f.endswith(".txt")]
            info[f"{split}_labels"] = len(lbls)
        else:
            info[f"{split}_labels"] = 0

    aug_policy = DATASET_DIR / "augmentation" / "augmentation_policy.json"
    aug_yaml = DATASET_DIR / "augmentation" / "yolo_training_augmentation_config.yaml"
    info["augmentation_policy_exists"] = aug_policy.exists()
    info["augmentation_yaml_exists"] = aug_yaml.exists()

    vm = DATASET_DIR / "version_manifest.json"
    info["version_manifest_exists"] = vm.exists()
    if vm.exists():
        import json
        with open(vm, encoding="utf-8") as f:
            v = json.load(f)
        info["dataset_version"] = v.get("version_id", "UNKNOWN")
    return info


def check_disk() -> dict:
    usage = shutil.disk_usage(REPO_ROOT)
    return {
        "disk_total_gb": round(usage.total / 1024**3, 1),
        "disk_free_gb": round(usage.free / 1024**3, 1),
    }


def check_pretrained() -> dict:
    info: dict = {}
    try:
        from ultralytics import checks
        info["yolov8n_downloadable"] = True
    except Exception:
        info["yolov8n_downloadable"] = "UNKNOWN (ultralytics not importable)"

    historical = REPO_ROOT / "backend" / "app" / "model" / "best.pt"
    info["historical_checkpoint_found"] = historical.exists()
    if historical.exists():
        info["historical_checkpoint_WARNING"] = (
            "DO NOT USE — trained on previous leaky dataset. "
            "Use official yolov8n.pt only."
        )
    return info


def generate_report(results: dict) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "# Phase 6.1 — Environment Compatibility Report",
        f"\nGenerated: {now}\n",
    ]

    sys_info = results["system"]
    lines.append("## System")
    lines.append(f"- OS: {sys_info['os']}")
    lines.append(f"- Architecture: {sys_info['arch']}")
    lines.append(f"- Python: {sys_info['python_version']}")
    lines.append(f"- Interpreter: {sys_info['python_path']}")

    nv = results["nvidia"]
    lines.append("\n## NVIDIA GPU")
    lines.append(f"- GPU: {nv.get('gpu_name', 'UNKNOWN')}")
    lines.append(f"- VRAM Total: {nv.get('vram_total_mb', 0)} MB")
    lines.append(f"- VRAM Free: {nv.get('vram_free_mb', 0)} MB")
    lines.append(f"- Driver: {nv.get('driver_version', 'UNKNOWN')}")

    cu = results["cuda"]
    lines.append("\n## CUDA / PyTorch")
    lines.append(f"- PyTorch: {cu.get('torch_version', 'N/A')}")
    lines.append(f"- CUDA Available: {cu.get('cuda_available', False)}")
    lines.append(f"- CUDA Build: {cu.get('cuda_build_version', 'N/A')}")
    lines.append(f"- Device Name: {cu.get('cuda_device_name', 'N/A')}")
    lines.append(f"- Torch VRAM: {cu.get('torch_vram_total_mb', 0)} MB")

    status = "PASS" if cu.get("cuda_available") else "FAIL — CUDA not available"
    lines.append(f"\n**CUDA Status: {status}**")

    ul = results["ultralytics"]
    lines.append("\n## Ultralytics")
    lines.append(f"- Importable: {ul.get('ultralytics_importable', False)}")
    lines.append(f"- Version: {ul.get('ultralytics_version', 'N/A')}")

    lines.append("\n## Python Packages")
    for pkg, ver in results["packages"].items():
        flag = " [MISSING]" if ver == "NOT INSTALLED" else ""
        lines.append(f"- {pkg}: {ver}{flag}")

    ds = results["dataset"]
    lines.append("\n## Dataset")
    lines.append(f"- Directory exists: {ds.get('dataset_dir_exists')}")
    lines.append(f"- data.yaml exists: {ds.get('data_yaml_exists')}")
    lines.append(f"- Version: {ds.get('dataset_version', 'N/A')}")
    lines.append(f"- Train images: {ds.get('train_images', 0)}")
    lines.append(f"- Valid images: {ds.get('valid_images', 0)}")
    lines.append(f"- Test images: {ds.get('test_images', 0)}")
    lines.append(f"- Augmentation policy: {ds.get('augmentation_policy_exists')}")
    lines.append(f"- Augmentation YAML: {ds.get('augmentation_yaml_exists')}")

    dk = results["disk"]
    lines.append("\n## Disk")
    lines.append(f"- Total: {dk['disk_total_gb']} GB")
    lines.append(f"- Free: {dk['disk_free_gb']} GB")

    pt = results["pretrained"]
    lines.append("\n## Pretrained Checkpoints")
    if pt.get("historical_checkpoint_found"):
        lines.append(f"- **WARNING:** Historical best.pt found — DO NOT USE for Phase 6.")
    lines.append(f"- Official yolov8n.pt: will be downloaded on first training run")

    critical = []
    if not cu.get("cuda_available"):
        critical.append("CUDA is NOT available — GPU training will fail.")
    if not ul.get("ultralytics_importable"):
        critical.append("Ultralytics is NOT installed.")
    if not ds.get("dataset_dir_exists"):
        critical.append("Training dataset directory not found.")
    if dk["disk_free_gb"] < 5:
        critical.append(f"Low disk space: {dk['disk_free_gb']} GB free.")

    if critical:
        lines.append("\n## CRITICAL ISSUES")
        for c in critical:
            lines.append(f"- {c}")
        lines.append("\n**Overall: NOT READY — resolve issues above before training.**")
    else:
        lines.append("\n## Overall Status")
        lines.append("**READY — environment supports YOLOv8n GPU training.**")

    return "\n".join(lines)


def main():
    print("Phase 6.1 — Environment Verification")
    print("=" * 50)

    results = {}
    print("Checking system...")
    results["system"] = check_system()
    print("Checking NVIDIA GPU...")
    results["nvidia"] = check_nvidia()
    print("Checking CUDA / PyTorch...")
    results["cuda"] = check_cuda()
    print("Checking packages...")
    results["packages"] = check_packages()
    print("Checking Ultralytics...")
    results["ultralytics"] = check_ultralytics_compat()
    print("Checking dataset...")
    results["dataset"] = check_dataset()
    print("Checking disk space...")
    results["disk"] = check_disk()
    print("Checking pretrained checkpoints...")
    results["pretrained"] = check_pretrained()

    report = generate_report(results)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "environment_report.md"
    report_path.write_text(report, encoding="utf-8")

    print(f"\nReport saved: {report_path}")
    print("\n" + report)
    return results


if __name__ == "__main__":
    main()
