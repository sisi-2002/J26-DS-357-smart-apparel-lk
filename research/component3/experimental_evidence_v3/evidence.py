"""Frozen-model evidence interventions; no labels used in feature construction."""
import sys
from pathlib import Path
from hashlib import sha256
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
for directory in ['scripts', 'experimental_multihorizon', 'experimental_multihorizon_v2', 'experimental_multihorizon_v3']:
    sys.path.insert(0, str(ROOT / 'research/component3' / directory))
sys.dont_write_bytecode = True
import build_and_evaluate as original
import model_inference as inference
from hazard_v2 import outcomes
from methods_v3 import UnpooledHazard

HORIZONS = (7, 14, 30)
KEYS = ['machineID', 'prediction_time']
FEATURES = inference.FEATURE_COLUMNS
EVIDENCE = {'H': 'observed_maintenance_span_days', 'M': 'maintenance_events_prior',
            'F': 'failure_events_prior', 'E': 'error_events_prior',
            'R': 'days_since_last_maintenance', 'ER': 'days_since_last_error',
            'C': 'published_field_completeness_pct'}
MAINTENANCE = [c for c in FEATURES if c.startswith('maintenance_events_') or c == 'days_since_last_maintenance']
ERRORS = [c for c in FEATURES if c.startswith('error_events_') or c == 'days_since_last_error']
RECENCIES = [c for c in FEATURES if c.startswith('days_since_last_')]
SEEDS = (11, 29, 47, 71, 97)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def conditions():
    result = [dict(condition='full', family='reference', streams=[], days=None, removal=0., seed=None, mask=[])]
    for streams, name in [([1, 2, 3], 'all'), ([1], 'maintenance'), ([3], 'errors')]:
        for days in (180, 90):
            result.append(dict(condition=f'{name}_{days}d', family='truncation', streams=streams,
                               days=days, removal=0., seed=None, mask=[]))
    for stream, name in [(1, 'maintenance'), (3, 'errors')]:
        for removal in (.25, .5, .75):
            for seed in SEEDS:
                result.append(dict(condition=f'{name}_drop{int(removal*100)}_s{seed}', family='thinning',
                                   streams=[stream], days=None, removal=removal, seed=seed, mask=[]))
    for name, columns in [('maintenance', MAINTENANCE), ('errors', ERRORS), ('recencies', RECENCIES)]:
        result.append(dict(condition=f'mask_{name}', family='masking', streams=[], days=None,
                           removal=0., seed=None, mask=columns))
    return result


def stable_uniform(source, seed):
    """Persistent record-level deletion keys; independent of future rows and labels."""
    category = next(c for c in ['comp', 'errorID', 'failure'] if c in source)
    keys = source[['machineID', 'datetime', category]].copy()
    occurrence = keys.groupby(list(keys), dropna=False).cumcount()
    return pd.Series([int.from_bytes(sha256(f'{seed}|{mid}|{pd.Timestamp(t).isoformat()}|{cat}|{n}'.encode()).digest()[:8], 'big') / 2**64
                      for (mid, t, cat), n in zip(keys.itertuples(index=False, name=None), occurrence)], index=source.index)


def historical_sources(sources, timestamp, spec, uniforms=None):
    """Future rows never enter an intervention. Unaltered sources remain read-only."""
    t = pd.Timestamp(timestamp)
    require(t.tzinfo is None and not pd.isna(t), 'Naive historical time required')
    result = [sources[0]]
    for stream, source in enumerate(sources[1:], 1):
        selected = source.datetime < t
        if stream in spec['streams']:
            if spec['days'] is not None:
                selected &= source.datetime >= t - pd.Timedelta(days=spec['days'])
            if spec['removal']:
                u = uniforms[stream] if uniforms is not None else stable_uniform(source, spec['seed'])
                selected &= u >= spec['removal']
        result.append(source.loc[selected].copy())
    return tuple(result)


def reconstruct(sources, landmarks, spec):
    require(not landmarks.duplicated(KEYS).any(), 'Duplicate landmarks')
    uniforms = {stream: stable_uniform(sources[stream], spec['seed']) for stream in spec['streams']} if spec['removal'] else None
    chunks = []
    for t, group in landmarks.groupby('prediction_time', sort=False):
        past = historical_sources(sources, t, spec, uniforms)
        rows = [inference.machine_features(*past, int(mid), t) for mid in group.machineID]
        chunk = pd.concat(rows, ignore_index=True)
        chunk.index = group.index
        chunks.append(chunk)
    frame = pd.concat(chunks).reindex(landmarks.index)
    require(all(c in frame for c in FEATURES), 'Incomplete inference schema')
    for name, column in [('maintenance', 'maintenance_events_prior'), ('failure', 'failure_events_prior'), ('error', 'error_events_prior')]:
        frame['available_' + name + '_records'] = frame[column]
    # Missing feature values are handled only by the already fitted imputer.
    for column in spec['mask']:
        require(column in FEATURES and column != 'machine_model', 'Only fitted numeric fields support masking')
        frame[column] = np.nan
    return frame


def validate_probabilities(p, count):
    p = np.asarray(p)
    require(p.shape == (count, 3) and np.isfinite(p).all(), 'Invalid probability matrix')
    require(((p >= 0) & (p <= 1)).all() and (np.diff(p, axis=1) >= 0).all(), 'Invalid cumulative ordering')
    return p


def verify_splits(panel):
    splits = {s: panel[panel.split.eq(s)].copy() for s in ['train', 'validation', 'test']}
    require([len(f) for f in splits.values()] == [1800, 500, 1300], 'Changed original cohort')
    for a, b in [('train', 'validation'), ('validation', 'test')]:
        require(splits[a].prediction_time.max() + pd.Timedelta(days=30) < splits[b].prediction_time.min(), '30-day embargo violated')
    return splits


def secondary_feasibility(splits):
    """No fitting: enforce availability of independent, fully purged error targets."""
    earliest = splits['validation'].prediction_time.max() + pd.Timedelta(days=30)
    latest = splits['test'].prediction_time.min() - pd.Timedelta(days=30)
    candidates = pd.date_range(earliest.normalize(), latest.normalize(), freq='D') if earliest < latest else []
    return dict(status='not_estimable_for_frozen_release', eligible_days=len(candidates),
                secondary_time_must_be_after=str(earliest), secondary_time_must_be_before=str(latest),
                train_cases=len(splits['train']), calibration_cases=len(splits['validation']), test_cases=len(splits['test']),
                reason='Training predictions are in-sample; calibration labels fitted the frozen sigmoid calibrators; test labels cannot train secondary models. No independent earlier purged period remains.')
