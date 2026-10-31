"""Research-only next-recorded-failure hazard model with closed upper bounds."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experimental_multihorizon'))
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from hazard_model import (PooledHazardModel, FEATURES, HORIZONS, landmark_outcomes,
                          cumulative_probabilities, require, make_baseline)


def load_research_model(path):
    bundle = joblib.load(path)
    require(isinstance(bundle, dict) and bundle.get('schema') == 'c3-hazard-v2-inclusive-research',
            'Incompatible research artifact schema')
    require(bundle.get('features') == FEATURES and isinstance(bundle.get('model'), HazardV2),
            'Incompatible research feature/model schema')
    return bundle['model']


def outcomes(snapshots, failures, observation_end):
    """Exclusive administrative follow-up; events at t excluded, at t+h included."""
    out = landmark_outcomes(snapshots, failures, observation_end)
    out['complete_30d'] = out.followup_days > 30
    for h in HORIZONS:
        out[f'label_failure_{h}d'] = np.where(
            out.first_event_days.notna() & out.first_event_days.le(h), 1.,
            np.where(out.followup_days.gt(h), 0., np.nan))
    return out


def periods(out):
    require(out.landmark_position.is_unique, 'Duplicate landmark positions')
    require((out.followup_days > 0).all(), 'Follow-up must be positive')
    seen = out.first_event_days.notna()
    require((np.isfinite(out.loc[seen, 'first_event_days']) &
             out.loc[seen, 'first_event_days'].gt(0) &
             out.loc[seen, 'first_event_days'].lt(out.loc[seen, 'followup_days'])).all(),
            'Event must precede exclusive censoring end')
    rows = []
    for row in out.itertuples(index=False):
        for interval, upper in enumerate(HORIZONS, 1):
            event = pd.notna(row.first_event_days) and row.first_event_days <= upper
            if event or row.followup_days > upper:
                rows.append((row.landmark_position, interval, int(event)))
            else:
                break
            if event:
                break
    return pd.DataFrame(rows, columns=['landmark_position', 'interval', 'event'])


class HazardV2(PooledHazardModel):
    """18 frozen historical features + interval identity and duration; one classifier.

    Inherits strict schema checks and frozen validation-only sigmoid calibration.
    Preprocessing fits unique training landmarks before person-period expansion.
    """
    def __init__(self, algorithm='lr', parameters=None):
        self.algorithm = algorithm
        self.parameters = parameters

    def _design(self, features, pp):
        base = super()._design(features, pp)
        duration = np.array([7., 7., 16.])[pp.interval.to_numpy(dtype=int) - 1] / 30.
        return np.column_stack([base, duration])

    def fit(self, train_features, train_periods):
        features = self._features(train_features)
        self._validate_periods(train_periods, len(features))
        require(train_periods.event.nunique() == 2, 'Training requires both outcomes')
        self.preprocessor_ = clone(make_baseline().named_steps['transform'])
        self.preprocessor_.fit(features)
        self.interval_encoder_ = OneHotEncoder(categories=[[1, 2, 3]], drop='first', sparse_output=False)
        self.interval_encoder_.fit(train_periods[['interval']])
        if self.algorithm == 'lr':
            self.classifier_ = LogisticRegression(max_iter=2000, random_state=42)
        elif self.algorithm == 'hgb':
            self.classifier_ = HistGradientBoostingClassifier(
                random_state=42, early_stopping=False, **(self.parameters or {}))
        else:
            raise ValueError('Unknown algorithm')
        self.classifier_.fit(self._design(features, train_periods), train_periods.event.to_numpy())
        self.feature_schema_ = list(FEATURES)
        self.training_landmarks_ = len(features)
        self.training_periods_ = len(train_periods)
        if hasattr(self, 'calibrator_'):
            del self.calibrator_
        return self
