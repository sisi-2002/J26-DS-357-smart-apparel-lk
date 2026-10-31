#!/usr/bin/env python3
"""Preliminary N2 controlled history-window pilot (7-day failure target).

Three training/evaluation conditions share identical machines, snapshots,
chronological splits, and future labels. Every model is REFIT separately using
features with only the stated quantity of historical event data available.
This is not a machine-specific prediction reliability formula, nor a final N2
relationship analysis. Machine-event sufficiency and real missingness cannot be
validated using this synthetic benchmark alone.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from build_and_evaluate import (
    load_csv, build_panel, FEATURE_NUMERIC, FEATURE_CAT, make_baseline,
    select_threshold, safe_score
)

def main(data_dir:Path,out_dir:Path):
    out_dir.mkdir(parents=True,exist_ok=True)
    machines=load_csv(data_dir,'machines',False)
    maint=load_csv(data_dir,'maint',True)
    failures=load_csv(data_dir,'failures',True)
    errors=load_csv(data_dir,'errors',True)
    records=[]; scored=[]
    target='label_failure_7d'
    ref_keys=None; ref_labels=None
    for name,days in [('Full observed history',None),('180-day history',180),('90-day history',90)]:
        panel=build_panel(machines,maint,failures,errors,max_history_days=days)
        # labels and samples are fixed across ablation scenarios
        keys=panel[['machineID','prediction_time','split']].to_records(index=False)
        labels=panel[target].to_numpy()
        if ref_keys is not None:
            assert np.array_equal(keys,ref_keys), 'experimental subjects changed'
            assert np.array_equal(labels,ref_labels), 'target changed across history conditions'
        else:
            ref_keys=keys; ref_labels=labels
        splits={s:panel[panel.split==s] for s in ['train','validation','test']}
        train,val,test=(splits[s] for s in ['train','validation','test'])
        xcols=FEATURE_NUMERIC+FEATURE_CAT
        model=make_baseline()
        model.fit(train[xcols],train[target])
        pv=model.predict_proba(val[xcols])[:,1]
        pt=model.predict_proba(test[xcols])[:,1]
        threshold=select_threshold(val[target],pv)
        res=safe_score(test[target],pt,threshold)
        res['history_condition']=name
        res['trained_separately']=True
        res['median_maint_count_on_test']=float(test['maintenance_events_prior'].median())
        res['median_failures_prior_on_test']=float(test['failure_events_prior'].median())
        res['test_machine_snapshots']=len(test)
        records.append(res)
        frame=test[['machineID','prediction_time',target]].copy()
        frame['history_condition']=name
        frame['predicted_probability']=pt
        frame['sq_error']=(frame[target]-pt)**2
        scored.append(frame)
    pd.DataFrame(records).to_csv(out_dir/'n2_initial_window_comparison.csv',index=False)
    pd.concat(scored,ignore_index=True).to_csv(out_dir/'n2_paired_test_prediction_losses.csv',index=False)
    print(pd.DataFrame(records).to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=Path('raw'))
    p.add_argument('--out-dir',type=Path,default=Path('outputs'))
    args=p.parse_args()
    main(args.data_dir,args.out_dir)
