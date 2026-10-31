#!/usr/bin/env python3
"""Compare sigmoid calibration methods for C3 on historical Azure PdM data.

Run from the corrected PP1 project root:
    python C3_calibration_validation_experiment.py --data-dir raw --out-dir outputs_calibration

Put this file in the project root, next to scripts/ and raw/.
The 3-fold weighted-CV option below is DIAGNOSTIC ONLY: its CV split is not
chronological and should not be presented as the selected research protocol.
The strictly later validation period is used for the primary sigmoid calibration.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import average_precision_score, brier_score_loss

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'scripts'))
from build_and_evaluate import (  # noqa: E402
    load_csv, build_panel, FEATURE_NUMERIC, FEATURE_CAT, HORIZONS,
    make_baseline,
)


def main(data_dir: Path, out_dir: Path):
    machines = load_csv(data_dir, 'machines', False)
    maintenance = load_csv(data_dir, 'maint', True)
    failures = load_csv(data_dir, 'failures', True)
    errors = load_csv(data_dir, 'errors', True)
    panel = build_panel(machines, maintenance, failures, errors)
    train = panel.loc[panel['split'] == 'train']
    validation = panel.loc[panel['split'] == 'validation']
    test = panel.loc[panel['split'] == 'test']
    columns = FEATURE_NUMERIC + FEATURE_CAT
    rows = []

    for horizon in HORIZONS:
        label = f'label_failure_{horizon}d'
        ytrain = train[label].to_numpy()
        ytest = test[label].to_numpy()
        prior = float(np.mean(ytrain))

        # Existing baseline: preprocessing and LogisticRegression from the corrected code.
        base = make_baseline()
        base.fit(train[columns], ytrain)
        p_uncalibrated = base.predict_proba(test[columns])[:, 1]

        # Chronological calibration: model trained on TRAIN, sigmoid fitted on later VALIDATION.
        # FrozenEstimator ensures the existing base model is NOT refitted on validation.
        calibrated = CalibratedClassifierCV(FrozenEstimator(base), method='sigmoid')
        calibrated.fit(validation[columns], validation[label])
        p_chronological = calibrated.predict_proba(test[columns])[:, 1]

        # Diagnostic comparison with the advice under review: balanced logistic regression,
        # 3-fold stratified CV calibration. Folds do NOT preserve chronological separation.
        weighted = make_baseline()
        weighted.set_params(logistic__class_weight='balanced')
        naive_cv3 = CalibratedClassifierCV(weighted, method='sigmoid', cv=3)
        naive_cv3.fit(train[columns], ytrain)
        p_cv3 = naive_cv3.predict_proba(test[columns])[:, 1]

        for name, probabilities in [
            ('train_prevalence_baseline', np.full(len(ytest), prior)),
            ('uncalibrated_original', p_uncalibrated),
            ('balanced_stratified_cv3_diagnostic_only', p_cv3),
            ('sigmoid_calibrated_on_later_validation', p_chronological),
        ]:
            rows.append({
                'horizon_days': horizon,
                'method': name,
                'test_samples': len(ytest),
                'test_positives': int(ytest.sum()),
                'brier_score': round(float(brier_score_loss(ytest, probabilities)), 6),
                'pr_auc_average_precision': round(float(average_precision_score(ytest, probabilities)), 6),
            })
    result = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / 'calibration_brier_comparison.csv'
    result.to_csv(dest, index=False)
    print(result.to_string(index=False))
    print(f'\nSaved: {dest}')
    print('WARNING: exploratory benchmark result, not apparel-machine deployment validation.')
    print('WARNING: stratified cv=3 is diagnostic only; it does not respect chronological ordering.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('raw'))
    parser.add_argument('--out-dir', type=Path, default=Path('outputs_calibration'))
    args = parser.parse_args()
    main(args.data_dir, args.out_dir)
