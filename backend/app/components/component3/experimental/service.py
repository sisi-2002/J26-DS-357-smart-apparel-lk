"""Load the audited v3 fitted estimators; inference never builds labels or fits."""
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import sys
from threading import RLock

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from app.core.config import REPOSITORY_ROOT, Settings
from ..ml.failure_inference import get_research_inference, _load_module, _lock
from ..ml.failure_model_loader import _validate_sources


class ExperimentalPredictionFailed(Exception):
    pass


def _require(value, message):
    if not value:
        raise ValueError(message)


def _load_bundle(payload):
    """Scope flat research imports to loading; restore the existing app registry."""
    root = REPOSITORY_ROOT / 'research/component3'
    files = {
        'build_and_evaluate': root / 'scripts/build_and_evaluate.py',
        'hazard_model': root / 'experimental_multihorizon/hazard_model.py',
        'hazard_v2': root / 'experimental_multihorizon_v2/hazard_v2.py',
        'methods_v3': root / 'experimental_multihorizon_v3/methods_v3.py',
    }
    with _lock:
        previous = {name: sys.modules.get(name) for name in files}
        paths, bytecode = sys.path[:], sys.dont_write_bytecode
        try:
            sys.dont_write_bytecode = True
            for name, path in files.items():
                sys.modules[name] = _load_module(name, path)
            bundle = joblib.load(BytesIO(payload))
            model_class = sys.modules['methods_v3'].UnpooledHazard
            return bundle, model_class
        finally:
            sys.path[:] = paths
            sys.dont_write_bytecode = bytecode
            for name, old in previous.items():
                if old is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = old


class ExperimentalPredictionService:
    @classmethod
    def load(cls, settings: Settings):
        _require(settings.c3_experimental_model_path is not None and settings.c3_data_dir is not None,
                 'Configure experimental artifact and benchmark data')
        manifest = json.loads(Path(__file__).with_name('v3_manifest.json').read_text())
        _require(sklearn.__version__ == manifest['sklearn'], 'Audited sklearn version required')
        # Check trusted artifact bytes BEFORE deserialization; no arbitrary model fallback.
        payload = settings.c3_experimental_model_path.read_bytes()
        digest = sha256(payload).hexdigest()
        _require(digest == manifest['artifact_sha256'], 'Experimental artifact identity mismatch')
        for relative, expected in manifest['source_sha256'].items():
            _require(sha256((REPOSITORY_ROOT / relative).read_bytes()).hexdigest() == expected,
                     'Experimental inference source mismatch')
        for filename, expected in manifest['data_sha256'].items():
            _require(sha256((settings.c3_data_dir / filename).read_bytes()).hexdigest() == expected,
                     'Experimental benchmark data mismatch')
        inference = get_research_inference()
        bundle, model_class = _load_bundle(payload)
        _require(isinstance(bundle, dict) and bundle.get('schema') == 'unpooled-inclusive-v3', 'Invalid experimental schema')
        _require(bundle.get('features') == inference.FEATURE_COLUMNS, 'Experimental feature schema mismatch')
        model = bundle.get('model')
        _require(isinstance(model, model_class) and model.algorithm == 'hgb', 'Expected unpooled HGB artifact')
        _require(model.feature_schema_ == inference.FEATURE_COLUMNS and len(model.models_) == len(model.calibrators_) == 3,
                 'Incomplete fitted model')
        for base, calibrated in zip(model.models_, model.calibrators_):
            _require(isinstance(calibrated, CalibratedClassifierCV) and calibrated.method == 'sigmoid'
                     and isinstance(calibrated.estimator, FrozenEstimator)
                     and calibrated.estimator.estimator is base, 'Expected frozen conditional sigmoid calibration')
            _require(list(base.feature_names_in_) == inference.FEATURE_COLUMNS, 'Preprocessing schema mismatch')
        sources = inference.read_demo_sources(settings.c3_data_dir)
        _validate_sources(sources)
        _require(set(sources[0].machineID) == set(range(1, 101)), 'Expected all benchmark machines')
        service = cls()
        service.model, service.sources, service.inference = model, sources, inference
        service.model_version = 'sha256:' + digest
        service.release = manifest['release']
        service._batch_cache = {}
        service._batch_lock = RLock()
        # Exercise fitted transformations and all three calibrated estimators.
        service.predict(1, '2015-09-02T06:00:00')
        return service

    def predict(self, machine_id: int, timestamp: str):
        if machine_id not in set(self.sources[0].machineID):
            raise KeyError('Unknown benchmark machine')
        try:
            features = self.inference.machine_features(*self.sources, machine_id, timestamp)
            p = self._probabilities(features, 1)
        except Exception as exc:
            raise ExperimentalPredictionFailed('Experimental inference failed') from exc
        return dict(machine_id=machine_id, prediction_time=timestamp,
                    p7=float(p[0, 0]), p14=float(p[0, 1]), p30=float(p[0, 2]),
                    evidence_profile=self._evidence(features.iloc[0]), **self.metadata())

    def _probabilities(self, features, count):
        p = np.asarray(self.model.predict(features, calibrated=True), dtype=float)
        _require(p.shape == (count, 3) and np.isfinite(p).all(), 'Invalid model output')
        _require(((p >= 0) & (p <= 1)).all() and (np.diff(p, axis=1) >= 0).all(), 'Incoherent model output')
        return p

    @staticmethod
    def _evidence(row):
        def number(key):
            return None if pd.isna(row[key]) else float(row[key])
        return dict(H_observed_maintenance_span_days=number('observed_maintenance_span_days'),
                    M_prior_maintenance_events=int(row.maintenance_events_prior),
                    F_prior_failure_events=int(row.failure_events_prior),
                    R_days_since_latest_maintenance=number('days_since_last_maintenance'),
                    C_published_field_completeness_pct=100.0,
                    error_events_prior=int(row.error_events_prior), error_events_7d=int(row.error_events_7d),
                    error_events_30d=int(row.error_events_30d), error_events_90d=int(row.error_events_90d),
                    days_since_last_error=number('days_since_last_error'))

    def rank(self, timestamp, horizon):
        try:
            _require(horizon in (7, 14, 30), 'Unsupported horizon')
            # Canonical bounded keys: never cache arbitrary dates or partial benchmarks.
            t = pd.Timestamp(timestamp)
            _require(t in self.inference.valid_demo_timestamps(), 'Unsupported historical time')
            key = t.isoformat()
            with self._batch_lock:
                if key not in self._batch_cache:
                    ids = self.sources[0].machineID.tolist()
                    _require(len(ids) == 100 and set(ids) == set(range(1, 101)), 'Incomplete benchmark')
                    features = pd.concat([self.inference.machine_features(*self.sources, mid, key)
                                          for mid in range(1, 101)], ignore_index=True)
                    p = self._probabilities(features, 100)
                    self._batch_cache[key] = tuple((mid, *map(float, values))
                                                  for mid, values in zip(range(1, 101), p))
                rows = self._batch_cache[key]
            column = (7, 14, 30).index(horizon) + 1
            ordered = sorted(rows, key=lambda row: (-row[column], row[0]))[:10]
            machines = [dict(rank=i, machine_id=row[0], p7=row[1], p14=row[2], p30=row[3])
                        for i, row in enumerate(ordered, 1)]
            return dict(prediction_time=key, horizon_days=horizon, scored_machine_count=100,
                        returned_machine_count=10, machines=machines, **self.metadata())
        except Exception as exc:
            raise ExperimentalPredictionFailed('Experimental batch inference failed') from exc

    def metadata(self):
        return dict(model_version=self.model_version, model_release=self.release,
                    model_schema='unpooled-inclusive-v3', experimental=True, research_only=True,
                    data_status='synthetic_azure_historical_demo',
                    interval_semantics='(0,7], (7,14], (14,30]; features strictly before prediction time',
                    note='Synthetic Azure 2015 benchmark; exploratory, repeatedly inspected test data. '
                         'Not validated for apparel factories or maintenance alerts. All horizons use the same '
                         'v3 conditional hazard system; they do not use the deployed seven-day LR. '
                         'Thirty-day Top-10 ranking was weaker than the independent LR research reference.')
