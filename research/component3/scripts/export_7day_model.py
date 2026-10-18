#!/usr/bin/env python3
"""Train once; export a locally trusted, fitted, calibrated C3 7-day model.

Run from research/component3:
 python scripts/export_7day_model.py --data-dir "PATH_TO_RAW" --model-out models/c3_7day_sigmoid.joblib
"""
from __future__ import annotations
import argparse
from pathlib import Path
import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from build_and_evaluate import build_panel, make_baseline
from model_inference import SCHEMA_VERSION, FEATURE_COLUMNS, read_demo_sources


def export_model(data_dir: Path, model_out: Path) -> Path:
    machines, maint, failures, errors = read_demo_sources(data_dir)
    panel = build_panel(machines, maint, failures, errors)
    train = panel.loc[panel.split == 'train']
    validation = panel.loc[panel.split == 'validation']
    label = 'label_failure_7d'
    if train[label].nunique() != 2 or validation[label].nunique() != 2:
        raise ValueError('Need both failure and non-failure examples for training/calibration')

    base = make_baseline()
    base.fit(train[FEATURE_COLUMNS], train[label].to_numpy())

    # Calibrator is fitted on LATER validation records, not random CV folds.
    # FrozenEstimator keeps the base pipeline frozen during calibration.
    calibrated = CalibratedClassifierCV(FrozenEstimator(base), method='sigmoid')
    calibrated.fit(validation[FEATURE_COLUMNS], validation[label].to_numpy())

    data = {
        'schema_version': SCHEMA_VERSION,
        'horizon_days': 7,
        'feature_columns': FEATURE_COLUMNS,
        'model': calibrated,
        'training_snapshots': int(len(train)),
        'calibration_snapshots': int(len(validation)),
        'dataset': 'Microsoft Azure synthetic predictive maintenance benchmark',
        'label_interval': '(prediction_time, prediction_time + 7 days)',
        'evaluation_warning': 'The test period has been inspected in earlier C3 experiments; do not treat it as untouched final test.',
    }
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(data, model_out)
    print('Saved calibrated 7-day model:', model_out)
    print('Training examples:', len(train), '| Calibration examples:', len(validation))
    print('Model training complete. The prediction CLI loads this file without retraining.')
    print('Only load this locally generated joblib file; never load untrusted model files.')
    return model_out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', type=Path, required=True)
    ap.add_argument('--model-out', type=Path, default=Path('models/c3_7day_sigmoid.joblib'))
    a = ap.parse_args()
    export_model(a.data_dir, a.model_out)
