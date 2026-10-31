"""Phase 6.3 — Training pipeline tests.

Tests configuration, validation, dataset access, and safety guards
WITHOUT running any model training or inference.
"""

import csv
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

TRAINING_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = TRAINING_DIR.parent.parent.parent
DATASET_ROOT = REPO_ROOT / "data-preprocessing" / "master_fabric_dataset" / "processed" / "chenab_detection_training_v1"
CONFIGS_DIR = TRAINING_DIR / "configs"

sys.path.insert(0, str(TRAINING_DIR))


# --- Config loading ---
class TestConfigLoading:
    def test_e01_loads(self):
        from train_yolov8n import load_config
        cfg = load_config(CONFIGS_DIR / "e01_baseline.yaml")
        assert cfg["experiment_id"] == "E01_baseline"

    def test_e02_loads(self):
        from train_yolov8n import load_config
        cfg = load_config(CONFIGS_DIR / "e02_optimized.yaml")
        assert cfg["experiment_id"] == "E02_optimized"

    def test_e03_loads(self):
        from train_yolov8n import load_config
        cfg = load_config(CONFIGS_DIR / "e03_high_resolution.yaml")
        assert cfg["experiment_id"] == "E03_high_resolution"

    def test_all_configs_have_required_keys(self):
        from train_yolov8n import load_config, REQUIRED_KEYS
        for name in ("e01_baseline.yaml", "e02_optimized.yaml", "e03_high_resolution.yaml"):
            cfg = load_config(CONFIGS_DIR / name)
            for key in REQUIRED_KEYS:
                assert key in cfg, f"{name} missing key: {key}"


# --- Config validation ---
class TestConfigValidation:
    def test_e01_valid(self):
        from train_yolov8n import load_config, validate_config
        cfg = load_config(CONFIGS_DIR / "e01_baseline.yaml")
        errors = validate_config(cfg)
        assert len(errors) == 0, errors

    def test_e02_valid(self):
        from train_yolov8n import load_config, validate_config
        cfg = load_config(CONFIGS_DIR / "e02_optimized.yaml")
        errors = validate_config(cfg)
        assert len(errors) == 0, errors

    def test_e03_valid(self):
        from train_yolov8n import load_config, validate_config
        cfg = load_config(CONFIGS_DIR / "e03_high_resolution.yaml")
        errors = validate_config(cfg)
        assert len(errors) == 0, errors

    def test_rejects_wrong_model(self):
        from train_yolov8n import validate_config
        cfg = {"experiment_id": "test", "model": "yolov8s.pt", "imgsz": 640,
               "epochs": 10, "batch": 4, "optimizer": "AdamW", "lr0": 0.001,
               "patience": 10, "seed": 42}
        errors = validate_config(cfg)
        assert any("yolov8n.pt" in e for e in errors)

    def test_rejects_huge_imgsz(self):
        from train_yolov8n import validate_config
        cfg = {"experiment_id": "test", "model": "yolov8n.pt", "imgsz": 1280,
               "epochs": 10, "batch": 4, "optimizer": "AdamW", "lr0": 0.001,
               "patience": 10, "seed": 42}
        errors = validate_config(cfg)
        assert any("too large" in e.lower() for e in errors)

    def test_rejects_huge_batch(self):
        from train_yolov8n import validate_config
        cfg = {"experiment_id": "test", "model": "yolov8n.pt", "imgsz": 640,
               "epochs": 10, "batch": 32, "optimizer": "AdamW", "lr0": 0.001,
               "patience": 10, "seed": 42}
        errors = validate_config(cfg)
        assert any("batch" in e.lower() for e in errors)

    def test_rejects_excessive_epochs(self):
        from train_yolov8n import validate_config
        cfg = {"experiment_id": "test", "model": "yolov8n.pt", "imgsz": 640,
               "epochs": 1000, "batch": 4, "optimizer": "AdamW", "lr0": 0.001,
               "patience": 10, "seed": 42}
        errors = validate_config(cfg)
        assert any("epoch" in e.lower() for e in errors)


# --- Dataset validation ---
class TestDatasetValidation:
    def test_dataset_dir_exists(self):
        assert DATASET_ROOT.is_dir()

    def test_data_yaml_exists(self):
        assert (DATASET_ROOT / "data.yaml").exists()

    def test_train_images_count(self):
        imgs = [f for f in os.listdir(DATASET_ROOT / "train" / "images")
                if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        assert len(imgs) == 1900

    def test_valid_images_count(self):
        imgs = [f for f in os.listdir(DATASET_ROOT / "valid" / "images")
                if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        assert len(imgs) == 377

    def test_test_images_count(self):
        imgs = [f for f in os.listdir(DATASET_ROOT / "test" / "images")
                if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        assert len(imgs) == 378

    def test_data_yaml_seven_classes(self):
        text = (DATASET_ROOT / "data.yaml").read_text("utf-8")
        assert "nc: 7" in text

    def test_dataset_validate_function(self):
        from train_yolov8n import validate_dataset
        errors = validate_dataset()
        assert len(errors) == 0, errors

    def test_image_label_pairing(self):
        for split in ("train", "valid"):
            img_dir = DATASET_ROOT / split / "images"
            lbl_dir = DATASET_ROOT / split / "labels"
            img_stems = {os.path.splitext(f)[0] for f in os.listdir(img_dir)
                         if f.lower().endswith((".jpg", ".png", ".jpeg"))}
            lbl_stems = {os.path.splitext(f)[0] for f in os.listdir(lbl_dir)
                         if f.endswith(".txt")}
            assert img_stems == lbl_stems, f"{split} mismatch"


# --- Augmentation loading ---
class TestAugmentation:
    def test_policy_exists(self):
        assert (DATASET_ROOT / "augmentation" / "augmentation_policy.json").exists()

    def test_yaml_exists(self):
        assert (DATASET_ROOT / "augmentation" / "yolo_training_augmentation_config.yaml").exists()

    def test_load_augmentation_params(self):
        from train_yolov8n import load_augmentation_params
        params = load_augmentation_params()
        assert params["fliplr"] == 0.5
        assert params["flipud"] == 0.5
        assert params["degrees"] == 5.0
        assert params["hsv_h"] == 0.01
        assert params["shear"] == 0.0
        assert params["perspective"] == 0.0
        assert params["mixup"] == 0.0
        assert params["copy_paste"] == 0.0

    def test_disabled_transforms_zero(self):
        from train_yolov8n import load_augmentation_params
        params = load_augmentation_params()
        for key in ("shear", "perspective", "mixup", "copy_paste"):
            assert params[key] == 0.0, f"{key} should be 0.0"

    def test_conservative_hsv(self):
        from train_yolov8n import load_augmentation_params
        params = load_augmentation_params()
        assert params["hsv_h"] <= 0.015
        assert params["hsv_s"] <= 0.3
        assert params["hsv_v"] <= 0.3


# --- Train args ---
class TestTrainArgs:
    def test_build_train_args(self):
        from train_yolov8n import load_config, load_augmentation_params, build_train_args
        cfg = load_config(CONFIGS_DIR / "e01_baseline.yaml")
        aug = load_augmentation_params()
        data_yaml = DATASET_ROOT / "data.yaml"
        out = Path(tempfile.mkdtemp()) / "test_exp"
        args = build_train_args(cfg, aug, data_yaml, out)
        assert args["imgsz"] == 640
        assert args["epochs"] == 150
        assert args["batch"] == 4
        assert args["seed"] == 42
        assert args["amp"] is True
        assert args["shear"] == 0.0
        assert args["mixup"] == 0.0

    def test_augmentation_in_train_args(self):
        from train_yolov8n import load_config, load_augmentation_params, build_train_args
        cfg = load_config(CONFIGS_DIR / "e01_baseline.yaml")
        aug = load_augmentation_params()
        data_yaml = DATASET_ROOT / "data.yaml"
        out = Path(tempfile.mkdtemp()) / "test_exp"
        args = build_train_args(cfg, aug, data_yaml, out)
        assert args["hsv_h"] == 0.01
        assert args["degrees"] == 5.0
        assert args["fliplr"] == 0.5


# --- Output directory ---
class TestOutputDir:
    def test_get_output_dir_new(self):
        from train_yolov8n import get_output_dir
        out = get_output_dir("TEST_nonexistent_experiment_xyz")
        assert "TEST_nonexistent_experiment_xyz" in out.name

    def test_no_overwrite(self):
        from train_yolov8n import get_output_dir, RUNS_DIR
        test_dir = RUNS_DIR / "TEST_overwrite_check"
        test_dir.mkdir(parents=True, exist_ok=True)
        try:
            out = get_output_dir("TEST_overwrite_check")
            assert out != test_dir
            assert "_v2" in out.name
        finally:
            test_dir.rmdir()


# --- Metadata ---
class TestMetadata:
    def test_collect_metadata(self):
        from train_yolov8n import load_config, collect_metadata
        cfg = load_config(CONFIGS_DIR / "e01_baseline.yaml")
        out = Path(tempfile.mkdtemp()) / "test_meta"
        meta = collect_metadata(cfg, out)
        assert meta["experiment_id"] == "E01_baseline"
        assert meta["dataset"]["version"] == "chenab_detection_training_v1"
        assert "timestamp" in meta


# --- Safety guards ---
class TestSafety:
    def test_model_must_be_yolov8n(self):
        from train_yolov8n import validate_config
        for bad in ("yolov8s.pt", "yolov8m.pt", "best.pt", "last.pt"):
            cfg = {"experiment_id": "t", "model": bad, "imgsz": 640,
                   "epochs": 10, "batch": 4, "optimizer": "AdamW",
                   "lr0": 0.001, "patience": 10, "seed": 42}
            errors = validate_config(cfg)
            assert len(errors) > 0, f"Should reject model={bad}"

    def test_seed_42(self):
        from train_yolov8n import load_config
        for name in ("e01_baseline.yaml", "e02_optimized.yaml", "e03_high_resolution.yaml"):
            cfg = load_config(CONFIGS_DIR / name)
            assert cfg["seed"] == 42

    def test_all_use_official_pretrained(self):
        from train_yolov8n import load_config
        for name in ("e01_baseline.yaml", "e02_optimized.yaml", "e03_high_resolution.yaml"):
            cfg = load_config(CONFIGS_DIR / name)
            assert cfg["model"] == "yolov8n.pt"

    def test_version_manifest_intact(self):
        vm = DATASET_ROOT / "version_manifest.json"
        assert vm.exists()
        data = json.loads(vm.read_text("utf-8"))
        assert data["version_id"] == "chenab_detection_training_v1"
        assert data["test_split_locked"] is True


# --- Dry run ---
class TestDryRun:
    def test_dry_run_e01(self, capsys):
        from train_yolov8n import load_config, dry_run
        cfg = load_config(CONFIGS_DIR / "e01_baseline.yaml")
        result = dry_run(cfg)
        assert result is True
        captured = capsys.readouterr()
        assert "DRY RUN" in captured.out
        assert "[OK] Configuration valid" in captured.out

    def test_dry_run_e02(self, capsys):
        from train_yolov8n import load_config, dry_run
        cfg = load_config(CONFIGS_DIR / "e02_optimized.yaml")
        result = dry_run(cfg)
        assert result is True

    def test_dry_run_e03(self, capsys):
        from train_yolov8n import load_config, dry_run
        cfg = load_config(CONFIGS_DIR / "e03_high_resolution.yaml")
        result = dry_run(cfg)
        assert result is True


# --- Compare experiments (with fixtures) ---
class TestCompareExperiments:
    def test_find_no_experiments(self, tmp_path):
        from compare_experiments import find_experiments
        result = find_experiments(tmp_path / "nonexistent")
        assert result == []

    def test_find_experiment_with_metadata(self, tmp_path):
        from compare_experiments import find_experiments
        exp_dir = tmp_path / "E_test"
        exp_dir.mkdir()
        meta = {
            "experiment_id": "E_test",
            "config": {"imgsz": 640, "batch": 4},
            "training_duration_seconds": 100,
        }
        (exp_dir / "experiment_metadata.json").write_text(json.dumps(meta), encoding="utf-8")
        results = find_experiments(tmp_path)
        assert len(results) == 1
        assert results[0]["experiment_id"] == "E_test"

    def test_generate_comparison_empty(self, tmp_path):
        from compare_experiments import generate_comparison
        generate_comparison([], tmp_path / "out")

    def test_generate_comparison_with_data(self, tmp_path):
        from compare_experiments import generate_comparison
        experiments = [
            {"experiment_id": "E1", "config": {"imgsz": 640, "batch": 4, "epochs": 100,
             "optimizer": "AdamW", "lr0": 0.001}, "mAP50": 0.75, "mAP50_95": 0.55,
             "precision": 0.80, "recall": 0.70, "best_pt_size_mb": 6.2,
             "duration_seconds": 3600, "peak_gpu_mb": 3500},
            {"experiment_id": "E2", "config": {"imgsz": 640, "batch": 4, "epochs": 200,
             "optimizer": "AdamW", "lr0": 0.0005}, "mAP50": 0.78, "mAP50_95": 0.58,
             "precision": 0.82, "recall": 0.72, "best_pt_size_mb": 6.2,
             "duration_seconds": 5400, "peak_gpu_mb": 3500},
        ]
        out = tmp_path / "comparison"
        generate_comparison(experiments, out)
        assert (out / "experiment_comparison.csv").exists()
        assert (out / "experiment_comparison.md").exists()


# --- Select best model (with fixtures) ---
class TestSelectBest:
    def test_ranking(self):
        experiments = [
            {"experiment_id": "E1", "mAP50_95": 0.55, "mAP50": 0.75, "dir": "/tmp/e1"},
            {"experiment_id": "E2", "mAP50_95": 0.58, "mAP50": 0.78, "dir": "/tmp/e2"},
            {"experiment_id": "E3", "mAP50_95": 0.52, "mAP50": 0.72, "dir": "/tmp/e3"},
        ]
        ranked = sorted(
            [e for e in experiments if e.get("mAP50_95") is not None],
            key=lambda x: x["mAP50_95"],
            reverse=True,
        )
        assert ranked[0]["experiment_id"] == "E2"


# --- Phase 4 immutability ---
class TestPhase4Immutability:
    def test_xai_masks_v3_exists(self):
        v3 = REPO_ROOT / "data-preprocessing" / "master_fabric_dataset" / "research" / "chenab_xai_masks_v3"
        assert v3.is_dir()

    def test_xai_manifest_count(self):
        v3 = REPO_ROOT / "data-preprocessing" / "master_fabric_dataset" / "research" / "chenab_xai_masks_v3"
        with open(v3 / "manifests" / "final_mask_manifest.csv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 133

    def test_raw_not_modified(self):
        raw = REPO_ROOT / "data-preprocessing" / "master_fabric_dataset" / "raw"
        if raw.exists():
            for d in raw.rglob("*"):
                assert "phase6" not in str(d).lower()
                assert "training" not in d.name.lower() or "training_v1" not in d.name.lower()


# --- Environment checker ---
class TestEnvironmentChecker:
    def test_check_system(self):
        from check_environment import check_system
        info = check_system()
        assert "python_version" in info
        assert "os" in info

    def test_check_dataset(self):
        from check_environment import check_dataset
        info = check_dataset()
        assert info["dataset_dir_exists"] is True
        assert info["data_yaml_exists"] is True
        assert info["train_images"] == 1900
        assert info["valid_images"] == 377

    def test_check_disk(self):
        from check_environment import check_disk
        info = check_disk()
        assert info["disk_free_gb"] > 0


# --- Validate script guards ---
class TestValidateScript:
    def test_validate_script_exists(self):
        assert (TRAINING_DIR / "validate_yolov8n.py").exists()

    def test_compare_script_exists(self):
        assert (TRAINING_DIR / "compare_experiments.py").exists()

    def test_select_script_exists(self):
        assert (TRAINING_DIR / "select_best_model.py").exists()

    def test_check_env_script_exists(self):
        assert (TRAINING_DIR / "check_environment.py").exists()


# --- Fix: device=0 in train args ---
class TestDeviceSelection:
    def test_device_0_in_train_args(self):
        from train_yolov8n import load_config, load_augmentation_params, build_train_args
        cfg = load_config(CONFIGS_DIR / "e01_baseline.yaml")
        aug = load_augmentation_params()
        data_yaml = DATASET_ROOT / "data.yaml"
        out = Path(tempfile.mkdtemp()) / "test_dev"
        args = build_train_args(cfg, aug, data_yaml, out)
        assert args["device"] == 0

    def test_exist_ok_true(self):
        from train_yolov8n import load_config, load_augmentation_params, build_train_args
        cfg = load_config(CONFIGS_DIR / "e01_baseline.yaml")
        aug = load_augmentation_params()
        data_yaml = DATASET_ROOT / "data.yaml"
        out = Path(tempfile.mkdtemp()) / "test_exist"
        args = build_train_args(cfg, aug, data_yaml, out)
        assert args["exist_ok"] is True


# --- Fix: validate_yolov8n imgsz/batch params ---
class TestValidateParams:
    def test_validate_accepts_imgsz_batch(self):
        import inspect
        from validate_yolov8n import validate
        sig = inspect.signature(validate)
        assert "imgsz" in sig.parameters
        assert "batch" in sig.parameters
        assert sig.parameters["imgsz"].default == 640
        assert sig.parameters["batch"].default == 4


# --- Fix: incomplete experiments excluded ---
class TestIncompleteExperiments:
    def test_incomplete_excluded_from_selection(self, tmp_path):
        from compare_experiments import find_experiments
        exp_complete = tmp_path / "E_complete"
        exp_complete.mkdir()
        (exp_complete / "weights").mkdir()
        (exp_complete / "weights" / "best.pt").write_bytes(b"fake")
        (exp_complete / "experiment_metadata.json").write_text(
            json.dumps({"experiment_id": "E_complete", "config": {}}),
            encoding="utf-8",
        )
        exp_incomplete = tmp_path / "E_incomplete"
        exp_incomplete.mkdir()
        (exp_incomplete / "experiment_metadata.json").write_text(
            json.dumps({"experiment_id": "E_incomplete", "config": {}}),
            encoding="utf-8",
        )
        results = find_experiments(tmp_path)
        complete = [e for e in results if e.get("complete")]
        incomplete = [e for e in results if not e.get("complete")]
        assert len(complete) == 1
        assert complete[0]["experiment_id"] == "E_complete"
        assert len(incomplete) == 1
        assert incomplete[0]["experiment_id"] == "E_incomplete"

    def test_selection_needs_complete_and_scored(self, tmp_path):
        from compare_experiments import find_experiments
        exp = tmp_path / "E_no_best"
        exp.mkdir()
        (exp / "experiment_metadata.json").write_text(
            json.dumps({"experiment_id": "E_no_best", "config": {}}),
            encoding="utf-8",
        )
        results = find_experiments(tmp_path)
        scored_complete = [e for e in results
                          if e.get("mAP50_95") is not None and e.get("complete")]
        assert len(scored_complete) == 0


# --- Fix: resume validates experiment_id ---
class TestResumeSafety:
    def test_resume_requires_last_pt(self, tmp_path):
        resume_dir = tmp_path / "E_missing"
        resume_dir.mkdir()
        from train_yolov8n import load_config
        cfg = load_config(CONFIGS_DIR / "e01_baseline.yaml")
        with pytest.raises(SystemExit):
            import train_yolov8n
            train_yolov8n.resume_training(resume_dir, cfg)

    def test_resume_rejects_mismatched_experiment(self, tmp_path):
        resume_dir = tmp_path / "E_mismatch"
        resume_dir.mkdir()
        (resume_dir / "weights").mkdir()
        (resume_dir / "weights" / "last.pt").write_bytes(b"fake")
        (resume_dir / "experiment_metadata.json").write_text(
            json.dumps({"experiment_id": "E02_optimized"}),
            encoding="utf-8",
        )
        from train_yolov8n import load_config
        cfg = load_config(CONFIGS_DIR / "e01_baseline.yaml")
        with pytest.raises(SystemExit):
            import train_yolov8n
            train_yolov8n.resume_training(resume_dir, cfg)


# --- Fix: README paths ---
class TestReadmePaths:
    def test_readme_no_dotrun_typo(self):
        text = (TRAINING_DIR / "README_MANUAL_TRAINING.md").read_text("utf-8")
        assert ".runs" not in text

    def test_readme_correct_runs_path(self):
        text = (TRAINING_DIR / "README_MANUAL_TRAINING.md").read_text("utf-8")
        assert "runs/component1_training" in text or "runs\\component1_training" in text

    def test_readme_python_312(self):
        text = (TRAINING_DIR / "README_MANUAL_TRAINING.md").read_text("utf-8")
        assert "3.12" in text

    def test_readme_imgsz_flag(self):
        text = (TRAINING_DIR / "README_MANUAL_TRAINING.md").read_text("utf-8")
        assert "--imgsz" in text


# --- Fix: derived data.yaml for Ultralytics path resolution ---
class TestDerivedDataYaml:
    def test_create_derived_yaml(self, tmp_path):
        from train_yolov8n import create_derived_data_yaml, DATASET_ROOT
        derived = create_derived_data_yaml(tmp_path / "exp")
        assert derived.exists()
        text = derived.read_text("utf-8")
        ds_abs = str(DATASET_ROOT.resolve()).replace("\\", "/")
        assert f"path: {ds_abs}" in text
        assert "train: train/images" in text
        assert "val: valid/images" in text
        assert "test: test/images" in text
        assert "nc: 7" in text
        for name in ["baekra", "color issues", "contamination", "cut",
                      "gray stitch", "selvet", "stain"]:
            assert name in text

    def test_derived_yaml_has_absolute_path(self, tmp_path):
        from train_yolov8n import create_derived_data_yaml, DATASET_ROOT
        derived = create_derived_data_yaml(tmp_path / "exp2")
        text = derived.read_text("utf-8")
        for line in text.splitlines():
            if line.startswith("path:"):
                path_val = line.split(":", 1)[1].strip()
                assert os.path.isabs(path_val) or path_val[1] == ":"

    def test_derived_yaml_uses_forward_slashes(self, tmp_path):
        from train_yolov8n import create_derived_data_yaml
        derived = create_derived_data_yaml(tmp_path / "exp3")
        text = derived.read_text("utf-8")
        for line in text.splitlines():
            if line.startswith("path:"):
                path_val = line.split(":", 1)[1].strip()
                assert "\\" not in path_val

    def test_frozen_data_yaml_not_modified(self):
        text = (DATASET_ROOT / "data.yaml").read_text("utf-8")
        assert "path: ." in text

    def test_derived_yaml_in_train_args(self, tmp_path):
        from train_yolov8n import (load_config, load_augmentation_params,
                                    build_train_args, create_derived_data_yaml)
        cfg = load_config(CONFIGS_DIR / "e01_baseline.yaml")
        aug = load_augmentation_params()
        derived = create_derived_data_yaml(tmp_path / "exp_args")
        out = tmp_path / "out"
        args = build_train_args(cfg, aug, derived, out)
        assert args["data"] == str(derived)
        assert "data.yaml" in args["data"]


# --- Fix: path resolution from any working directory ---
class TestPathResolution:
    def test_dataset_root_is_absolute(self):
        from train_yolov8n import DATASET_ROOT
        assert DATASET_ROOT.is_absolute()

    def test_runs_dir_is_absolute(self):
        from train_yolov8n import RUNS_DIR
        assert RUNS_DIR.is_absolute()

    def test_dataset_splits_exist_absolute(self):
        from train_yolov8n import DATASET_ROOT
        for split in ("train", "valid", "test"):
            assert (DATASET_ROOT / split / "images").is_dir()
            assert (DATASET_ROOT / split / "labels").is_dir()

    def test_dry_run_from_different_cwd(self, tmp_path, capsys, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from train_yolov8n import load_config, dry_run
        cfg = load_config(CONFIGS_DIR / "e01_baseline.yaml")
        result = dry_run(cfg)
        assert result is True
        captured = capsys.readouterr()
        assert "[OK] Dataset paths resolve correctly" in captured.out


# --- Fix: failed E01 gets unique output dir ---
class TestFailedRunProvenance:
    def test_existing_e01_triggers_v2(self):
        from train_yolov8n import get_output_dir, RUNS_DIR
        e01_dir = RUNS_DIR / "E01_baseline"
        if e01_dir.exists():
            out = get_output_dir("E01_baseline")
            assert out != e01_dir
            assert "_v2" in out.name or "_v3" in out.name


# --- Validate script derived yaml ---
class TestValidateScriptDerivedYaml:
    def test_validate_creates_derived_yaml(self, tmp_path):
        from validate_yolov8n import create_derived_data_yaml, DATASET_ROOT
        derived = create_derived_data_yaml(tmp_path / "val_exp")
        assert derived.exists()
        text = derived.read_text("utf-8")
        ds_abs = str(DATASET_ROOT.resolve()).replace("\\", "/")
        assert f"path: {ds_abs}" in text
