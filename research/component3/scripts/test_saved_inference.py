#!/usr/bin/env python3
"""Offline tests of feature parity, strict-past boundaries and saved-model inference."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from build_and_evaluate import FEATURE_NUMERIC, FEATURE_CAT, build_panel, make_baseline
from model_inference import (
    FEATURE_COLUMNS, machine_features, read_demo_sources, predict_from_loaded,
    predict_saved, SCHEMA_VERSION,
)


def test(data_dir: Path, model_path: Path) -> None:
    sources = read_demo_sources(data_dir)
    t = pd.Timestamp('2015-09-02 06:00:00')
    mid = 1
    solo = machine_features(*sources, mid, t)
    assert list(solo[FEATURE_COLUMNS].columns) == FEATURE_COLUMNS

    # Independent parity check: single-snapshot features must exactly match
    # the known, original chronological-panel features for same machine/date.
    panel = build_panel(*sources)
    original = panel.loc[(panel.machineID == mid) & (panel.prediction_time == t)]
    assert len(original) == 1
    for col in FEATURE_COLUMNS:
        a, b = solo.iloc[0][col], original.iloc[0][col]
        assert (pd.isna(a) and pd.isna(b)) or a == b, f'Feature parity failed: {col}: {a}, {b}'
    print('PASS: single-machine features exactly match chronological training features')

    # Check maintenance, errors, failures at t and after t are not included.
    alt = [v.copy() for v in sources]
    for ix in (1, 2, 3):
        for new_t in (t, t + pd.Timedelta(days=3)):
            added = alt[ix].iloc[[0]].copy()
            added['machineID'] = mid
            added['datetime'] = new_t
            alt[ix] = pd.concat([alt[ix], added], ignore_index=True)
    altered = machine_features(*alt, mid, t)
    for col in FEATURE_COLUMNS:
        a, b = solo.iloc[0][col], altered.iloc[0][col]
        assert (pd.isna(a) and pd.isna(b)) or a == b, f'Temporal leakage detected: {col}'
    print('PASS: records at/after prediction time do not change model features')

    for invalid_id, invalid_time in [(999999, str(t)), (mid, '2026-10-18 06:00:00'), (mid, '2015-09-03 06:00:00')]:
        try:
            machine_features(*sources, invalid_id, invalid_time)
            raise AssertionError('Invalid demo input was accepted')
        except ValueError:
            pass
    print('PASS: unknown machines and non-demo timestamps rejected')

    saved = joblib.load(model_path)
    actual = predict_from_loaded(saved, sources, mid, str(t))
    again = predict_saved(model_path, data_dir, mid, str(t))
    assert actual == again
    assert 0 <= actual['predicted_failure_probability'] <= 1
    assert actual['research_only'] is True
    print('PASS: load saved model twice and return identical results without retraining')

    # Numerical agreement with the separate chronological calibration procedure.
    train = panel.loc[panel.split == 'train']
    val = panel.loc[panel.split == 'validation']
    fresh = make_baseline()
    fresh.fit(train[FEATURE_COLUMNS], train['label_failure_7d'].to_numpy())
    independently_calibrated = CalibratedClassifierCV(FrozenEstimator(fresh), method='sigmoid')
    independently_calibrated.fit(val[FEATURE_COLUMNS], val['label_failure_7d'].to_numpy())
    expected = independently_calibrated.predict_proba(solo[FEATURE_COLUMNS])[:, 1][0]
    stored = saved['model'].predict_proba(solo[FEATURE_COLUMNS])[:, 1][0]
    assert np.isclose(expected, stored, atol=1e-12), f'Calibration discrepancy {expected} != {stored}'
    print('PASS: saved calibrated probability matches independent chronological training')
    print('PASS: saved-model inference tests complete')


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir',type=Path,required=True)
    ap.add_argument('--model',type=Path,default=Path('models/c3_7day_sigmoid.joblib'))
    a=ap.parse_args()
    test(a.data_dir,a.model)
