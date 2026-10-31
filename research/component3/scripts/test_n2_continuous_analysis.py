#!/usr/bin/env python3
"""Targeted N2 checks. Can run basic unit tests without any raw CSVs.

Optional full validation:
  python scripts/test_n2_continuous_analysis.py --data-dir RAW \
    --losses outputs_corrected/n2_paired_test_prediction_losses.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from n2_continuous_analysis import (
    descriptive_associations, grouped_loss, grouped_machine_cv,
    load_and_link, TARGET, MODEL_INPUTS, CONDITIONS,
)
from build_and_evaluate import build_panel, load_csv, FEATURE_NUMERIC


def test_synthetic_relationship():
    rng=np.random.default_rng(2026)
    rows=[]
    for machine in range(1, 16):
        for week in range(7):
            m=int(rng.integers(1,20))
            r=int(rng.integers(1,20))
            rows.append({
                'machineID':machine,
                'prediction_time':pd.Timestamp('2015-09-02')+pd.Timedelta(days=7*week),
                'history_condition':'Full observed history',
                'machine_model':'modelA' if machine % 2 else 'modelB',
                'observed_maintenance_span_days': float(rng.integers(20,180)),
                'maintenance_events_prior':m,
                'failure_events_prior':int(rng.integers(0,8)),
                'days_since_last_maintenance':float(r),
                'published_field_completeness_pct':100.0,
                'sq_error':float(np.clip(0.025+0.003*r-0.0005*m+rng.normal(0,0.02),0,1)),
                TARGET:int(rng.integers(0,2)),
            })
    d=pd.DataFrame(rows)
    stats=descriptive_associations(d)
    assert pd.isna(stats.loc[stats.evidence_variable.eq('published_field_completeness_pct'),'spearman_rho_descriptive']).all()
    binned=grouped_loss(d,'maintenance_events_prior')
    assert int(binned.n_snapshots.sum())==len(d)
    summary,pred=grouped_machine_cv(d,folds=5)
    assert len(pred)==len(d)
    assert pred.crossfit_estimated_conditional_mean_loss.between(0,1).all()
    assert summary.n_machines.iloc[0]==15
    assert np.isfinite(summary.rmse_evidence_ridge).all()
    assert all(c in d.columns for c in MODEL_INPUTS)


def test_real_data(data_dir:Path, losses:Path):
    enriched=load_and_link(data_dir, losses)
    assert set(enriched.history_condition.unique())==set(CONDITIONS)
    assert not enriched.duplicated(['history_condition','machineID','prediction_time']).any()
    assert enriched['sq_error'].between(0,1).all()
    assert enriched.C_field_completeness_pct.eq(100).all()
    assert np.allclose(enriched.sq_error,(enriched[TARGET]-enriched.predicted_probability)**2)
    original_keys={}
    for name, group in enriched.groupby('history_condition'):
        original_keys[name]=set(zip(group.machineID,group.prediction_time))
    assert len({frozenset(v) for v in original_keys.values()})==1

    # Add events at EXACT prediction time and 35 days after. They must neither
    # enter the prior-history features nor the strict-future 7/14/30-day labels.
    machines=load_csv(data_dir,'machines',False)
    mid=int(machines.machineID.iloc[0]); machines=machines[machines.machineID.eq(mid)]
    maint=load_csv(data_dir,'maint',True); maint=maint[maint.machineID.eq(mid)].copy()
    failures=load_csv(data_dir,'failures',True); failures=failures[failures.machineID.eq(mid)].copy()
    errors=load_csv(data_dir,'errors',True); errors=errors[errors.machineID.eq(mid)].copy()
    t=pd.Timestamp('2015-09-02 06:00:00')
    control=build_panel(machines,maint,failures,errors)
    extra_dates=[t,t+pd.Timedelta(days=35)]
    append_maint=pd.DataFrame([{'machineID':mid,'datetime':tt,'comp':'comp1'} for tt in extra_dates])
    append_fail=pd.DataFrame([{'machineID':mid,'datetime':tt,'failure':'comp1'} for tt in extra_dates])
    append_err=pd.DataFrame([{'machineID':mid,'datetime':tt,'errorID':'error1'} for tt in extra_dates])
    altered=build_panel(machines,pd.concat([maint,append_maint]),pd.concat([failures,append_fail]),pd.concat([errors,append_err]))
    cols=FEATURE_NUMERIC + ['observed_maintenance_span_days','published_field_completeness_pct'] + [f'label_failure_{h}d' for h in (7,14,30)]
    a=control.loc[control.prediction_time.eq(t),cols].iloc[0]
    b=altered.loc[altered.prediction_time.eq(t),cols].iloc[0]
    assert np.allclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float), equal_nan=True), 'Past-only or strict-future boundary was violated'
    print('PASS: condition-specific evidence/labels/losses match saved files')
    print('PASS: events exactly at prediction time and >30d later do not leak into t')


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',type=Path)
    p.add_argument('--losses',type=Path)
    a=p.parse_args()
    test_synthetic_relationship()
    print('PASS: synthetic unit checks, including constant-completeness handling and grouped CV')
    if bool(a.data_dir) != bool(a.losses):
        p.error('For full checks, supply BOTH --data-dir and --losses')
    if a.data_dir:
        test_real_data(a.data_dir,a.losses)
    print('PASS: N2 continuous analysis tests')
