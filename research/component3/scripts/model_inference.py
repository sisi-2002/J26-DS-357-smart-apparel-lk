#!/usr/bin/env python3
"""Reusable, past-only, historical 7-day risk inference for C3 PP1.

This module intentionally DOES NOT create or read future failure labels and DOES
NOT retrain. It matches the baseline's feature specification exactly.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from build_and_evaluate import (
    FEATURE_CAT, FEATURE_NUMERIC, SPLIT_GRID, cut_stats, load_csv,
)

SCHEMA_VERSION = 'c3-pdm-7day-v1'
FEATURE_COLUMNS = FEATURE_NUMERIC + FEATURE_CAT


def valid_demo_timestamps() -> pd.DatetimeIndex:
    """Exactly the held-out timestamps previously used by the C3 baseline."""
    return pd.date_range(*SPLIT_GRID['test'], freq='7D')


def machine_features(
    machines: pd.DataFrame,
    maint: pd.DataFrame,
    failures: pd.DataFrame,
    errors: pd.DataFrame,
    machine_id: int,
    prediction_time: str | pd.Timestamp,
) -> pd.DataFrame:
    """Build one snapshot using ONLY events strictly before prediction_time.

    The four dataframes may contain events after prediction_time; these are not
    considered by cut_stats. No future labels are used or generated here.
    """
    t = pd.Timestamp(prediction_time)
    if pd.isna(t) or t.tzinfo is not None:
        raise ValueError('Prediction timestamp must be a timezone-naive valid historical time')
    if t not in valid_demo_timestamps():
        raise ValueError('For this PP1 demo, prediction_time must be one of the 13 original weekly 2015 test timestamps (Sep 2–Nov 25, 06:00)')

    if 'machineID' not in machines or 'model' not in machines or 'age' not in machines:
        raise ValueError('PdM_machines.csv is missing required machine fields')
    row = machines.loc[machines.machineID == machine_id]
    if len(row) != 1:
        raise ValueError(f'Machine ID {machine_id} not found uniquely in synthetic benchmark')
    m_row = row.iloc[0]

    def event_stats(source: pd.DataFrame, source_name: str):
        if not {'machineID', 'datetime'}.issubset(source.columns):
            raise ValueError(f'{source_name} missing machineID or datetime')
        event_times = pd.to_datetime(source.loc[source.machineID == machine_id, 'datetime'], errors='raise')
        if event_times.isna().any():
            raise ValueError(f'{source_name} has missing datetime values')
        values = np.sort(event_times.astype('int64').to_numpy())
        return cut_stats(values, t, None)

    mc, m7, m30, m90, m180, msince, mspan = event_stats(maint, 'maintenance')
    fc, _f7, f30, f90, f180, fsince, _fspan = event_stats(failures, 'failures')
    ec, e7, e30, e90, _e180, esince, _espan = event_stats(errors, 'errors')

    features = {
        'machine_model': str(m_row['model']),
        'machine_age_years': int(m_row['age']),
        'maintenance_events_prior': mc,
        'maintenance_events_7d': m7,
        'maintenance_events_30d': m30,
        'maintenance_events_90d': m90,
        'maintenance_events_180d': m180,
        'days_since_last_maintenance': msince,
        'failure_events_prior': fc,
        'failure_events_30d': f30,
        'failure_events_90d': f90,
        'failure_events_180d': f180,
        'days_since_last_failure': fsince,
        'error_events_prior': ec,
        'error_events_7d': e7,
        'error_events_30d': e30,
        'error_events_90d': e90,
        'days_since_last_error': esince,
        'observed_maintenance_span_days': mspan,
        'published_field_completeness_pct': 100.0,
    }
    return pd.DataFrame([features])


def read_demo_sources(data_dir: Path):
    return (
        load_csv(data_dir, 'machines', False),
        load_csv(data_dir, 'maint', True),
        load_csv(data_dir, 'failures', True),
        load_csv(data_dir, 'errors', True),
    )


def predict_from_loaded(
    saved: dict,
    sources: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
    machine_id: int,
    prediction_time: str,
) -> dict:
    if saved.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('Model schema version does not match inference script')
    if saved.get('feature_columns') != FEATURE_COLUMNS:
        raise ValueError('Model feature columns do not match the current feature specification')
    if saved.get('horizon_days') != 7:
        raise ValueError('Only the trained 7-day model is supported in this PP1 demo')
    x = machine_features(*sources, machine_id, prediction_time)
    p = float(saved['model'].predict_proba(x[FEATURE_COLUMNS])[:, 1][0])
    if not np.isfinite(p) or not (0 <= p <= 1):
        raise ValueError('Model produced an invalid probability')
    f = x.iloc[0]
    def nullable_number(v):
        return None if pd.isna(v) else float(v)
    return {
        'machine_id': int(machine_id),
        'prediction_time': str(pd.Timestamp(prediction_time)),
        'horizon_days': 7,
        'predicted_failure_probability': round(p, 6),
        'calibration': 'sigmoid fitted on later chronological validation period',
        'evidence_profile': {
            'H_observed_maintenance_span_days': nullable_number(f['observed_maintenance_span_days']),
            'M_prior_maintenance_events': int(f['maintenance_events_prior']),
            'F_prior_failure_events': int(f['failure_events_prior']),
            'R_days_since_latest_maintenance': nullable_number(f['days_since_last_maintenance']),
            'C_published_field_completeness_pct': 100.0,
        },
        'machine_type': str(f['machine_model']),
        'research_only': True,
        'note': 'Synthetic Azure benchmark; 2015 historical test snapshot. Not an apparel machine, real-time prediction, or validated maintenance decision. C=100% constant and H is not verified logging coverage.',
    }


def predict_saved(model_path: Path, data_dir: Path, machine_id: int, prediction_time: str) -> dict:
    if not model_path.is_file():
        raise FileNotFoundError(f'Model not found at {model_path}. Run export_7day_model.py first.')
    # SECURITY: Joblib uses pickle. Load ONLY the model YOU generated locally.
    saved = joblib.load(model_path)
    return predict_from_loaded(saved, read_demo_sources(data_dir), machine_id, prediction_time)


def format_result(result: dict) -> str:
    return json.dumps(result, indent=2, allow_nan=False)
