"""Authoritative, read-only path constants for Function 2 stage F2.1A.

All paths are relative to the repository root and were verified to exist on
2026-10-17 as part of F2.1A Task 1 (input verification). See
research/component1/function2/reports/task1_input_verification.md for the
verification procedure and evidence (checkpoint hash check, dataset counts).

Nothing in this module writes to, renames, or mutates any of these paths.
"""

from __future__ import annotations

from pathlib import Path

# research/component1/function2/config/paths.py -> repo root is 4 parents up
REPO_ROOT = Path(__file__).resolve().parents[4]

# --- Frozen Function 1 model (E02_optimized) ---------------------------------
FREEZE_MANIFEST = REPO_ROOT / "runs" / "component1_training" / "freeze" / "model_freeze_E02_optimized_v1.json"
E02_CHECKPOINT = REPO_ROOT / "runs" / "component1_training" / "E02_optimized" / "weights" / "best.pt"

# --- Chenab dataset (authoritative training dataset for the frozen model) ---
CHENAB_DATASET_ROOT = (
    REPO_ROOT / "data-preprocessing" / "master_fabric_dataset" / "processed"
    / "chenab_detection_training_v1"
)
CHENAB_SPLIT_MANIFEST = CHENAB_DATASET_ROOT / "split_manifest.csv"
CHENAB_TRAIN_MANIFEST = CHENAB_DATASET_ROOT / "train_manifest.csv"
CHENAB_DATA_YAML = CHENAB_DATASET_ROOT / "data.yaml"

CHENAB_TRAIN_IMAGES_DIR = CHENAB_DATASET_ROOT / "train" / "images"
CHENAB_TRAIN_LABELS_DIR = CHENAB_DATASET_ROOT / "train" / "labels"
# Validation/test directories are recorded for completeness only.
# Function 2 F2.1A code must NEVER open files under the test/ directory.
CHENAB_VALID_IMAGES_DIR = CHENAB_DATASET_ROOT / "valid" / "images"
CHENAB_VALID_LABELS_DIR = CHENAB_DATASET_ROOT / "valid" / "labels"
CHENAB_TEST_IMAGES_DIR = CHENAB_DATASET_ROOT / "test" / "images"  # LOCKED — do not open
CHENAB_TEST_LABELS_DIR = CHENAB_DATASET_ROOT / "test" / "labels"  # LOCKED — do not open

CLASS_NAMES = [
    "baekra", "color issues", "contamination", "cut",
    "gray stitch", "selvet", "stain",
]

# --- Shared, read-only modules (do not modify) --------------------------------
COMPLEXITY_FEATURES_MODULE = (
    REPO_ROOT / "research" / "component1" / "complexity_calibration" / "src"
    / "complexity_features.py"
)
DETECTION_XAI_MODULE = (
    REPO_ROOT / "research" / "component1" / "xai_experiments" / "src"
    / "detection_xai.py"
)
