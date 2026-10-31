"""F2.5 shared data-loading utilities.

Read-only access to F2.1-F2.4 protected outputs. This module never writes to
any path under results/xai_evaluation, results/xai_execution,
results/evaluation_sampling, results/calibration, results/reference_masks,
config/, or runs/ -- it only reads from them. All F2.5 outputs are written
by the calling scripts to
results/complexity_xai_analysis/f25_pp1_v1/.

No TEST-split access: every path referenced below is under VALID-derived
F2.1-F2.4 outputs (xai_primary_sample / valid_complexity_manifest / the
60-image XAI sample and its downstream manifests). This module never
imports or reads any held-out-split environment variable or manifest.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
F2_ROOT = REPO_ROOT / "research" / "component1" / "function2"

# ---- protected (read-only) input roots ----------------------------------
F23_FULL = F2_ROOT / "results" / "xai_execution" / "f23_full_run_v1"
F24_FULL = F2_ROOT / "results" / "xai_evaluation" / "f24_full_v1"
F24_PILOT = F2_ROOT / "results" / "xai_evaluation" / "f24_pilot_v1"
F22B = F2_ROOT / "results" / "evaluation_sampling" / "f22b_pp1_v1"
CALIBRATION = F2_ROOT / "results" / "calibration"
CONFIG = F2_ROOT / "config"
E02_CHECKPOINT = REPO_ROOT / "runs" / "component1_training" / "E02_optimized" / "weights" / "best.pt"

# ---- F2.5 output root (the only tree this stage writes to) --------------
F25_OUT = F2_ROOT / "results" / "complexity_xai_analysis" / "f25_pp1_v1"

SEED = 357


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def to_bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return None
    s = str(v).strip()
    if s == "True":
        return True
    if s == "False":
        return False
    if s == "":
        return None
    raise ValueError(f"not a bool: {v!r}")


def to_float(v, default=None):
    if v is None or str(v).strip() == "":
        return default
    return float(v)


def to_int(v, default=None):
    if v is None or str(v).strip() == "":
        return default
    return int(v)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=False, default=str)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ---- canonical source file paths (all read-only) -------------------------
PATHS = {
    "accounting": F24_FULL / "manifests" / "all_instances_accounting.csv",
    "metric_eligibility": F24_FULL / "manifests" / "metric_eligibility.csv",
    "sanity_subset": F24_FULL / "manifests" / "sanity_subset.csv",
    "localization": F24_FULL / "metrics" / "localization.csv",
    "faithfulness": F24_FULL / "metrics" / "faithfulness.csv",
    "stability_v1": F24_FULL / "metrics" / "stability.csv",
    "stability_v2": F24_FULL / "metrics" / "stability_v2_degenerate_flagged.csv",
    "sanity_classification_branch": F24_FULL / "sanity_checks" / "sanity_classification_branch.csv",
    "sanity_whole_head": F24_FULL / "sanity_checks" / "sanity_whole_head_diagnostic.csv",
    "timing_sample": F24_FULL / "timing" / "timing_sample.csv",
    "execution_manifest": F23_FULL / "manifests" / "execution_manifest.csv",
    "paired_explanations": F23_FULL / "manifests" / "paired_explanations_manifest.csv",
    "target_linkage": F23_FULL / "manifests" / "target_linkage.csv",
    "failures": F23_FULL / "manifests" / "failures.csv",
    "xai_primary_sample": F22B / "manifests" / "xai_primary_sample.csv",
    "valid_complexity_manifest": F22B / "manifests" / "valid_complexity_manifest.csv",
    "pixel_iou_pilot_subset": F22B / "manifests" / "pixel_iou_pilot_subset.csv",
    "pixel_iou_pilot_metric": F24_PILOT / "metrics" / "pixel_iou_pilot.csv",
    "weight_sensitivity_train": CALIBRATION / "weight_sensitivity_v2.csv",
    "f22b_task6_distribution_audit": F22B / "reports" / "f22b_task6_distribution_audit.json",
    "complexity_protocol_snapshot": CONFIG / "complexity_protocol_snapshot_pp1_v1.json",
    "f24b_config": CONFIG / "f24b_full_evaluation_config_v1.json",
    "f23b_config": CONFIG / "f23b_execution_config_v1.json",
}


def load_all():
    data = {}
    for key, path in PATHS.items():
        if not path.exists():
            data[key] = None
            continue
        if path.suffix == ".json":
            with open(path, encoding="utf-8") as fh:
                data[key] = json.load(fh)
        else:
            data[key] = read_csv(path)
    return data
