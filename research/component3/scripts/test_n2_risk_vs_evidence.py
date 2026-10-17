#!/usr/bin/env python3
"""Self-contained integrity tests for the N2 incremental-value experiment."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from n2_risk_vs_evidence import SPECS, run_experiment, validate_input


def synthetic_example():
    rows=[]
    for mid in range(1,11):
        for n in range(3):
            p=0.05+0.06*((mid+n)%9)
            y=int((mid+n)%4==0)
            rows.append({
              'machineID':mid,'prediction_time':str(pd.Timestamp('2015-09-02')+pd.Timedelta(days=7*n)),
              'history_condition':'Full observed history', 'machine_model':f'model{1+(mid%3)}',
              'predicted_probability':p,'label_failure_7d':y,'sq_error':(y-p)**2,
              'H_observed_span_days':150+mid*4+n*7,'M_maintenance_count':mid+n,
              'F_prior_failure_count':mid%3,'C_field_completeness_pct':100.0,
              'R_days_since_maintenance':n+mid%5,
            })
    return pd.DataFrame(rows)


def check():
    frame=synthetic_example()
    scores, predictions, deltas=run_experiment(frame, folds=5)
    assert len(scores)==len(SPECS)
    assert len(predictions)==len(frame)
    assert len(deltas)==1
    assert predictions['machine_group_fold'].between(1,5).all()
    for spec in SPECS:
        assert predictions[f'estimated_loss__{spec}'].between(0,1).all()
    # Every machine has exactly one fold across its repeated weekly snapshots.
    assert (predictions.groupby('machineID').machine_group_fold.nunique()==1).all()
    # Exact squared error integrity is checked before any modelling.
    wrong=frame.copy()
    wrong.loc[0,'sq_error']+=0.25
    try:
        validate_input(wrong)
    except ValueError as e:
        assert 'Loss values' in str(e)
    else:
        raise AssertionError('Failed to catch loss/label misalignment')
    # Intended no-leakage predictor sets must exclude observed loss and future target.
    for nums,cats in SPECS.values():
        assert 'sq_error' not in nums+cats
        assert 'label_failure_7d' not in nums+cats
        assert 'machineID' not in nums+cats
        assert 'C_field_completeness_pct' not in nums+cats
    print('PASS: synthetic data, squared-loss integrity, machine-grouped folds, and feature exclusions')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, help='Optional original N2 evidence/loss CSV for integrity checks')
    args=ap.parse_args()
    check()
    if args.input:
        frame=validate_input(pd.read_csv(args.input))
        print(f'PASS: source {len(frame)} rows, {frame.machineID.nunique()} unique machines, {frame.history_condition.nunique()} conditions')
        assert all(frame.groupby('history_condition').machineID.nunique()>=5)
    print('PASS: N2 incremental experiment tests')


if __name__=='__main__':
    main()
