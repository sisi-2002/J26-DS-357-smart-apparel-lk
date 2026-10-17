#!/usr/bin/env python3
"""C3 PP1: leakage-aware maintenance event classification baseline.

Input: the four raw CSVs PdM_machines.csv, PdM_maint.csv,
       PdM_failures.csv and PdM_errors.csv (synthetic Azure PdM).
Output: timestamped machine-horizon examples, prevalence and model reports.

IMPORTANT: actual maintenance logging coverage is not known. The earliest
recorded event is NOT proof that system logging began at that date.

This baseline is methodological validation on synthetic generic machines;
it does not demonstrate apparel-machinery predictive performance.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, precision_score,
    recall_score, f1_score, roc_auc_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HORIZONS=(7,14,30)
# Split by prediction time. A 30-day embargo allows train targets to finish
# before validation starts, and validation targets before test starts.
SPLIT_GRID={
    'train': ('2015-02-01 06:00:00','2015-06-01 06:00:00'),
    'validation': ('2015-07-02 06:00:00','2015-08-01 06:00:00'),
    'test': ('2015-09-02 06:00:00','2015-11-30 06:00:00'),
}
FEATURE_NUMERIC=[
    'machine_age_years',
    'maintenance_events_prior','maintenance_events_7d','maintenance_events_30d',
    'maintenance_events_90d','maintenance_events_180d',
    'failure_events_prior','failure_events_30d','failure_events_90d','failure_events_180d',
    'error_events_prior','error_events_7d','error_events_30d','error_events_90d',
    'days_since_last_maintenance','days_since_last_failure','days_since_last_error'
]
FEATURE_CAT=['machine_model']
SUFFICIENCY_COLS=[
    'observed_maintenance_span_days','maintenance_events_prior','failure_events_prior',
    'error_events_prior','days_since_last_maintenance',
    'maintenance_events_90d','published_field_completeness_pct'
]

def load_csv(path: Path, name: str, date: bool):
    src=path/f'PdM_{name}.csv'
    if not src.is_file():
        raise FileNotFoundError(f'Missing {src}; supply the 4 raw CSVs to --data-dir')
    return pd.read_csv(src, parse_dates=['datetime'] if date else None)

def cut_stats(events: np.ndarray, t: pd.Timestamp, max_history_days: int | None = None):
    """All event times in ns; counts from timestamps strictly BEFORE t."""
    now=t.value
    idx=int(np.searchsorted(events, now, side='left'))
    lower_cut=-np.inf if max_history_days is None else now-int(pd.Timedelta(days=max_history_days).value)
    start=0 if max_history_days is None else int(np.searchsorted(events, lower_cut, side='left'))
    past=events[start:idx]
    count=len(past)
    def n_days(days):
        earliest=max(lower_cut, now-int(pd.Timedelta(days=days).value))
        return int(idx-np.searchsorted(events, earliest, side='left'))
    since=np.nan if count==0 else (now-past[-1])/pd.Timedelta(days=1).value
    span=np.nan if count==0 else (now-past[0])/pd.Timedelta(days=1).value
    return count,n_days(7),n_days(30),n_days(90),n_days(180),since,span

def build_panel(machines, maint, failures, errors, max_history_days: int | None = None):
    # Basic structure guardrails.
    for name, df in [('maint',maint),('failures',failures),('errors',errors)]:
        assert df['datetime'].notna().all(),f'{name} has unparseable dates'
        assert set(df.machineID).issubset(set(machines.machineID)), f'{name} has unknown machine IDs'
    by_type={}
    for label,d in [('maint',maint),('failures',failures),('errors',errors)]:
        by_type[label]={int(k):np.sort(v.datetime.astype('int64').to_numpy()) for k,v in d.groupby('machineID')}

    rows=[]
    for split,(a,b) in SPLIT_GRID.items():
        # Weekly prediction snapshots for each machine, in this fixed date range
        times=pd.date_range(a,b, freq='7D')
        for t in times:
            for machine in machines.itertuples(index=False):
                mid=int(machine.machineID)
                m=by_type['maint'].get(mid,np.empty(0,dtype='int64'))
                f=by_type['failures'].get(mid,np.empty(0,dtype='int64'))
                e=by_type['errors'].get(mid,np.empty(0,dtype='int64'))
                mc,m7,m30,m90,m180,msince,mspan=cut_stats(m,t,max_history_days)
                fc,f7,f30,f90,f180,fsince,fspan=cut_stats(f,t,max_history_days)
                ec,e7,e30,e90,e180,esince,espan=cut_stats(e,t,max_history_days)
                row={
                    'split':split,'prediction_time':t,'machineID':mid,
                    'machine_model':str(machine.model), 'machine_age_years':int(machine.age),
                    'maintenance_events_prior':mc,'maintenance_events_7d':m7,
                    'maintenance_events_30d':m30,'maintenance_events_90d':m90,
                    'maintenance_events_180d':m180,'days_since_last_maintenance':msince,
                    'failure_events_prior':fc,'failure_events_30d':f30,
                    'failure_events_90d':f90,'failure_events_180d':f180,
                    'days_since_last_failure':fsince,
                    'error_events_prior':ec,'error_events_7d':e7,
                    'error_events_30d':e30,'error_events_90d':e90,
                    'days_since_last_error':esince,
                    # This is elapsed time since FIRST OBSERVED MAINTENANCE EVENT,
                    # NOT proven maintenance-log coverage or machine operation age.
                    'observed_maintenance_span_days':mspan,
                    # Published columns are complete in raw files; not field-rich CMMS.
                    'published_field_completeness_pct':100.,
                }
                for horizon in HORIZONS:
                    stop=(t+pd.Timedelta(days=horizon)).value
                    future_left=int(np.searchsorted(f,t.value,side='right'))
                    future_right=int(np.searchsorted(f,stop,side='left'))
                    row[f'label_failure_{horizon}d']=int(future_right>future_left)
                rows.append(row)
    panel=pd.DataFrame(rows)
    return panel

def safe_score(y, proba, threshold):
    y=np.asarray(y).astype(int)
    proba=np.asarray(proba)
    preds=(proba>=threshold).astype(int)
    return {
       'positive_cases':int(y.sum()),'sample_count':int(len(y)),
       'positive_prevalence':round(float(y.mean()),5),
       'pr_auc_average_precision':round(float(average_precision_score(y,proba)),5) if y.sum()>0 else None,
       'roc_auc':round(float(roc_auc_score(y,proba)),5) if len(np.unique(y))==2 else None,
       'brier_score':round(float(brier_score_loss(y,proba)),5),
       'precision':round(float(precision_score(y,preds,zero_division=0)),5),
       'recall':round(float(recall_score(y,preds,zero_division=0)),5),
       'f1':round(float(f1_score(y,preds,zero_division=0)),5),
       'threshold':float(threshold)
    }

def make_baseline():
    numeric=Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler())])
    cats=OneHotEncoder(handle_unknown='ignore')
    processor=ColumnTransformer([('num',numeric,FEATURE_NUMERIC),('cat',cats,FEATURE_CAT)])
    return Pipeline([('transform',processor),('logistic',LogisticRegression(max_iter=2000,random_state=42))])

def select_threshold(y_true, probs):
    # Select on validation ONLY; transparent, finite grid, optimise F1 for demo.
    # This is NOT a calibrated maintenance-action cost rule.
    options=np.linspace(0.05,0.95,91)
    vals=[f1_score(y_true,(probs>=th).astype(int),zero_division=0) for th in options]
    return round(float(options[int(np.argmax(vals))]),3)

def execute(data_dir:Path, out_dir:Path):
    out_dir.mkdir(parents=True,exist_ok=True)
    machines=load_csv(data_dir,'machines',False)
    maint=load_csv(data_dir,'maint',True)
    failures=load_csv(data_dir,'failures',True)
    errors=load_csv(data_dir,'errors',True)
    panel=build_panel(machines,maint,failures,errors)
    panel.to_csv(out_dir/'machine_prediction_examples.csv',index=False)
    # IMPORTANT: this CSV includes future labels for offline supervised evaluation.
    # Never feed the label_ columns as model features.
    summary=[]; evaluation=[]; sample_preds=[]
    for horizon in HORIZONS:
        label=f'label_failure_{horizon}d'
        for split in ['train','validation','test']:
            d=panel[panel.split==split]
            summary.append({
                'horizon_days':horizon,'split':split,'snapshots':d.prediction_time.nunique(),
                'machines':d.machineID.nunique(),'examples':len(d),
                'positive_examples':int(d[label].sum()),'negative_examples':int(len(d)-d[label].sum()),
                'positive_rate':round(float(d[label].mean()),4),
                'first_snapshot':str(d.prediction_time.min()),
                'last_snapshot':str(d.prediction_time.max()),
                'last_label_window_end':str(d.prediction_time.max()+pd.Timedelta(days=horizon)),
            })
        train=panel[panel.split=='train']; val=panel[panel.split=='validation']; test=panel[panel.split=='test']
        ytr=train[label].to_numpy(); yval=val[label].to_numpy(); ytest=test[label].to_numpy()
        if not (len(np.unique(ytr))==2 and len(np.unique(yval))==2 and len(np.unique(ytest))==2):
            print(f'SKIP horizon={horizon}: insufficient positive/negative events in split')
            continue
        # No future label columns are included in the feature lists.
        columns=FEATURE_NUMERIC+FEATURE_CAT
        lr=make_baseline()
        lr.fit(train[columns],ytr)
        vp=lr.predict_proba(val[columns])[:,1]
        tp=lr.predict_proba(test[columns])[:,1]
        threshold=select_threshold(yval,vp)
        dummy=DummyClassifier(strategy='prior')
        dummy.fit(train[columns],ytr)
        for model_name, model_p, mt in [('logistic_regression',tp,threshold),('train_prevalence_baseline',dummy.predict_proba(test[columns])[:,1],0.5)]:
            metrics=safe_score(ytest,model_p,mt)
            evaluation.append({'horizon_days':horizon,'split':'test','model':model_name,
                               'threshold_source':'validation_f1' if model_name=='logistic_regression' else 'not_tuned',**metrics})
        prediction=test[['machineID','machine_model','prediction_time','split',label,*SUFFICIENCY_COLS]].copy()
        prediction['horizon_days']=horizon
        prediction['predicted_failure_probability']=tp
        prediction['prediction_error_sq']=(ytest-tp)**2
        prediction['alert_at_validation_threshold']=(tp>=threshold).astype(int)
        sample_preds.append(prediction)
        print(f'Finished horizon {horizon} days: {len(train)} training, {len(val)} validation, {len(test)} test rows')
    pd.DataFrame(summary).to_csv(out_dir/'horizon_split_label_audit.csv',index=False)
    pd.DataFrame(evaluation).to_csv(out_dir/'baseline_model_metrics.csv',index=False)
    predictions=pd.concat(sample_preds,ignore_index=True)
    predictions.to_csv(out_dir/'heldout_predictions_with_evidence.csv',index=False)
    # Assert chronological target separation; can use longest 30d to cover all.
    times=panel.groupby('split').prediction_time.agg(['min','max'])
    assert times.loc['train','max']+pd.Timedelta(days=30) < times.loc['validation','min'], 'train -> val label overlap'
    assert times.loc['validation','max']+pd.Timedelta(days=30) < times.loc['test','min'], 'val -> test label overlap'
    # Validate observed date max fully includes all test labels; this is not proof of
    # comprehensive logging, merely a check that timestamps are within source span.
    max_target=times.loc['test','max']+pd.Timedelta(days=30)
    assert max_target < pd.Timestamp('2016-01-01 06:00:00'), 'test label window ends after reporting interval'
    audit={
       'dataset':'Synthetic Azure Predictive Maintenance; generic models, not apparel machines',
       'machine_count':len(machines),'maintenance_rows':len(maint),'failure_rows':len(failures),'error_rows':len(errors),
       'feature_time_rule':'only timestamps strictly before prediction_time',
       'target_time_rule':'failure timestamps >= prediction_time and < prediction_time+horizon',
       'temporal_split':'weekly snapshots; train Feb-Jun, val July, test Sep-Nov 2015; >=30d target embargo',
       'prediction_horizons':list(HORIZONS),
       'observed_history_warning':'observed_maintenance_span_days is elapsed since first logged event, NOT verified maintenance system coverage',
       'maintenance_warning':'maintenance rows include corrective replacements; no PM/CM or downtime data',
       'data_quality_warning':'published available fields 100 percent complete; simulated masking needed to test missingness, cannot validate real missingness mechanism',
       'evaluation_warning':'repeated samples from same 100 machines across time; labels for horizons overlap within each split; not independent rows; results are exploratory',
       'dataset_imbalance_warning':'must calculate label prevalence per HORIZON, not from original failure row count divided by total maintenance rows',
       'max_test_horizon_end':str(max_target),
       'input_feature_columns':FEATURE_NUMERIC+FEATURE_CAT,
       'audit_only_sufficiency_columns':SUFFICIENCY_COLS,
       'outputs':['machine_prediction_examples.csv','horizon_split_label_audit.csv','baseline_model_metrics.csv','heldout_predictions_with_evidence.csv'],
    }
    (out_dir/'run_summary.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(pd.DataFrame(summary).to_string(index=False))
    print(pd.DataFrame(evaluation).to_string(index=False))
    return audit

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=Path('raw'))
    parser.add_argument('--out-dir',type=Path,default=Path('outputs'))
    args=parser.parse_args()
    execute(args.data_dir,args.out_dir)
