"""Isolated, chronological, inclusive-boundary pooled-hazard v2 comparison."""
import os
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '1'
import sys
sys.dont_write_bytecode = True
import argparse
import importlib.util
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import roc_auc_score, precision_recall_curve
from hazard_v2 import HazardV2, FEATURES, HORIZONS, outcomes, periods, require, load_research_model
from train_evaluate import (feasibility, independent_references, load_csv, make_baseline,
                            evaluate, ordering, sha256)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experimental_algorithm_comparison'))
from models import development_folds

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
GRID = [dict(max_leaf_nodes=7, max_depth=3, min_samples_leaf=40, l2_regularization=3., learning_rate=.05, max_iter=100),
        dict(max_leaf_nodes=15, max_depth=4, min_samples_leaf=40, l2_regularization=5., learning_rate=.05, max_iter=150),
        dict(max_leaf_nodes=7, max_depth=3, min_samples_leaf=60, l2_regularization=10., learning_rate=.03, max_iter=150)]


def top10(frame, y, p, method, horizon):
    rows = []
    for date, g in frame.assign(y=y, p=p).groupby('prediction_time'):
        ranked = g.sort_values(['p', 'machineID'], ascending=[False, True]).head(10)
        captured = int(ranked.y.sum())
        rows.append(dict(method=method, horizon=horizon, prediction_time=str(date), positives=int(g.y.sum()),
                         captured=captured, precision_at_10=captured/10,
                         recall_at_10=captured/g.y.sum() if g.y.sum() else np.nan,
                         expected_random_captured=10*g.y.mean()))
    return rows


def metric(frame, y, p, method, h):
    row, bins = evaluate(y, p, method, h)
    ranks = top10(frame, y, p, method, h)
    captured = sum(r['captured'] for r in ranks)
    row.update(roc_auc=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,
               precision_at_10=captured/(10*len(ranks)), recall_at_10=captured/int(y.sum()) if y.sum() else None,
               top10_captured=captured, random_top10_expected=sum(r['expected_random_captured'] for r in ranks))
    return row, bins, ranks


def run(data_dir, run_dir):
    require(run_dir.resolve().is_relative_to(HERE) and not run_dir.exists(), 'Use a NEW run directory inside v2; never overwrite results')
    run_dir.mkdir(parents=True)
    original_artifact = RESEARCH/'models/c3_7day_sigmoid.joblib'
    protected_hash = sha256(original_artifact)
    sources = tuple(load_csv(data_dir, n, n!='machines') for n in ['machines','maint','failures','errors'])
    audit = {}
    for name, source in zip(['machines','maint','failures','errors'], sources):
        require(not source.isna().any().any(), f'Missing raw fields in {name}')
        require(source.machineID.isin(range(1,101)).all(), 'Unexpected raw machine ID')
        audit[name] = dict(rows=len(source), columns=list(source), duplicates=int(source.duplicated().sum()),
                           sha256=sha256(data_dir/f'PdM_{name}.csv'))
        if name!='machines':
            audit[name].update(first=str(source.datetime.min()), last=str(source.datetime.max()),
                               categories=sorted(source.iloc[:,-1].astype(str).unique().tolist()))
    require(sources[0].machineID.is_unique and len(sources[0])==100, 'Machine schema mismatch')
    splits, old_out, _, old_counts = feasibility(sources, '2016-01-01')
    raw_ref, old_ref, fidelity = independent_references(*[splits[s] for s in ['train','validation','test']])
    require(ordering(old_ref)['either']==272, 'Legacy calibrated reference audit mismatch')
    old_algo = pd.read_csv(RESEARCH/'experimental_algorithm_comparison/results/model_metrics.csv')
    old_algo.to_csv(run_dir/'legacy_algorithm_reference.csv',index=False)
    new_out, pp, boundary = {}, {}, []
    for split, frame in splits.items():
        new_out[split] = outcomes(frame, sources[2], '2016-01-01')
        require(new_out[split].complete_30d.all(), 'Incomplete cohort follow-up; no cases silently dropped')
        for h in HORIZONS:
            boundary.append(dict(split=split,horizon=h,changed_labels=int((new_out[split][f'label_failure_{h}d'].to_numpy()!=frame[f'label_failure_{h}d'].to_numpy()).sum())))
            frame[f'label_failure_{h}d'] = new_out[split][f'label_failure_{h}d'].to_numpy(dtype=int)
        pp[split] = periods(new_out[split])
    train, val, test = [splits[s] for s in ['train','validation','test']]
    # Fixed plan written before candidate fitting and before any new test scores.
    plan=dict(grid=GRID, selection='mean uncalibrated cumulative Brier across three horizons and three TRAIN-only purged folds',
              features=FEATURES, calibration='one pooled sigmoid fitted on later validation risk sets, base frozen',
              observation_end_exclusive='2016-01-01', intervals=['(0,7]','(7,14]','(14,30]'])
    (run_dir/'plan.json').write_text(json.dumps(plan,indent=2))
    dev=[]
    for candidate, parameters in enumerate(GRID):
        for fitting, scoring, info in development_folds(train):
            a,b=train.iloc[fitting],train.iloc[scoring]
            model=HazardV2('hgb',parameters).fit(a, periods(outcomes(a,sources[2],'2016-01-01')))
            p=model.predict(b)
            for j,h in enumerate(HORIZONS):
                dev.append(dict(candidate=candidate, **info, horizon=h,
                                brier=float(np.mean((b[f'label_failure_{h}d'].to_numpy()-p[:,j])**2))))
    development=pd.DataFrame(dev)
    winner=int(development.groupby('candidate').brier.mean().sort_values(kind='stable').index[0])
    development.to_csv(run_dir/'development.csv',index=False)
    predictions={s:{} for s in splits}; runtime=[]; models={}
    for algorithm in ['lr','hgb']:
        start=time.perf_counter()
        model=HazardV2(algorithm,GRID[winner] if algorithm=='hgb' else None).fit(train,pp['train'])
        for s,frame in splits.items(): predictions[s][algorithm+'_raw']=model.predict(frame)
        model.calibrate(val,pp['validation'])
        for s,frame in splits.items(): predictions[s][algorithm+'_sigmoid']=model.predict(frame, calibrated=True)
        models[algorithm]=model
        infer=time.perf_counter(); model.predict(test,calibrated=True); inference_s=time.perf_counter()-infer
        runtime.append(dict(model=algorithm,fit_calibration_seconds=time.perf_counter()-start,
                            batch_1300_inference_seconds=inference_s,serialized_bytes=len(__import__('pickle').dumps(model))))
    # Fresh same-cohort/same-target independent LR comparator, not the deployed artifact.
    references={s:[] for s in splits}
    for h in HORIZONS:
        target=f'label_failure_{h}d'
        base=make_baseline().fit(train[FEATURES],train[target])
        calibrated=CalibratedClassifierCV(FrozenEstimator(base),method='sigmoid').fit(val[FEATURES],val[target])
        for s,frame in splits.items(): references[s].append(calibrated.predict_proba(frame[FEATURES])[:,1])
    for s,frame in splits.items():
        predictions[s]['independent_sigmoid']=np.column_stack(references[s])
        predictions[s]['train_prevalence']=np.tile(train[[f'label_failure_{h}d' for h in HORIZONS]].mean().to_numpy(),(len(frame),1))
    metrics=[]; bins=[]; ranks=[]; coherences={}; details=[]
    for s,frame in splits.items():
        y=frame[[f'label_failure_{h}d' for h in HORIZONS]].to_numpy()
        for method,p in predictions[s].items():
            coherences[s+'/'+method]=ordering(p)
            if method.startswith(('lr_','hgb_')): require(ordering(p)['either']==0,'Coherence failure')
            for j,h in enumerate(HORIZONS):
                row,bb,rr=metric(frame,y[:,j],p[:,j],method,h)
                metrics.append(dict(split=s,**row)); bins.extend([dict(split=s,**r) for r in bb])
                if s=='test': ranks.extend(rr)
                if s=='test':
                    detail=frame[['machineID','prediction_time']].copy()
                    detail['method']=method; detail['horizon']=h; detail['y']=y[:,j]; detail['p']=p[:,j]
                    details.append(detail)
    metrics=pd.DataFrame(metrics); metrics.to_csv(run_dir/'metrics_all_splits.csv',index=False,float_format='%.17g')
    metrics[metrics.split.eq('test')].to_csv(run_dir/'comparison.csv',index=False,float_format='%.17g')
    pd.DataFrame(bins).to_csv(run_dir/'calibration_bins.csv',index=False)
    pd.DataFrame(ranks).to_csv(run_dir/'top10_by_date.csv',index=False)
    pd.concat(details).to_csv(run_dir/'heldout_predictions.csv',index=False,float_format='%.17g')
    pd.DataFrame(boundary).to_csv(run_dir/'boundary_differences.csv',index=False)
    pd.DataFrame(runtime).to_csv(run_dir/'runtime.csv',index=False)
    # Paired cluster resampling of FIXED predictions, not refits; two dependence views.
    deltas=[]; yy=test[[f'label_failure_{h}d' for h in HORIZONS]].to_numpy()
    ref_loss=(yy-predictions['test']['independent_sigmoid'])**2
    for cluster in ['machineID','prediction_time']:
        labels=test[cluster].to_numpy(); groups=np.unique(labels); rng=np.random.default_rng(42)
        draws=rng.integers(0,len(groups),(1000,len(groups)))
        for method in ['lr_sigmoid','hgb_sigmoid']:
            diff=(yy-predictions['test'][method])**2-ref_loss
            means=np.array([diff[labels==g].mean(axis=0) for g in groups]); sampled=means[draws].mean(axis=1)
            for j,h in enumerate(HORIZONS): deltas.append(dict(cluster=cluster,method=method,horizon=h,delta=diff[:,j].mean(),
                low=np.quantile(sampled[:,j],.025),high=np.quantile(sampled[:,j],.975)))
    pd.DataFrame(deltas).to_csv(run_dir/'paired_cluster_deltas.csv',index=False)
    shift=[]
    for column in FEATURES[:-1]:
        for s,frame in splits.items(): shift.append(dict(feature=column,split=s,mean=frame[column].mean(),
            median=frame[column].median(),missing=int(frame[column].isna().sum())))
    pd.DataFrame(shift).to_csv(run_dir/'feature_shift.csv',index=False)
    require(sha256(original_artifact)==protected_hash,'Original artifact changed')
    summary=dict(dataset='Synthetic Azure industrial machines; NOT validated apparel-factory data',
        sklearn=sklearn.__version__,numpy=np.__version__,pandas=pd.__version__,python=sys.version,
        source_audit=audit,legacy_reference_fidelity=fidelity,
        legacy_ordering=ordering(old_ref),legacy_split_counts=old_counts,
        selected_hgb_candidate=winner,selected_hgb_parameters=GRID[winner],ordering=coherences,
        risk_sets={s:pp[s].groupby('interval').event.agg(['count','sum']).reset_index().to_dict('records') for s in pp},
        original_artifact_sha256=protected_hash,sksurv_available=importlib.util.find_spec('sksurv') is not None,
        saved_artifacts=[],integration='not implemented; requires review', seed=42)
    # Research bundles only. No change to deployed artifact or application imports.
    for algorithm,model in models.items():
        artifact=run_dir/f'hazard_v2_{algorithm}_research.joblib'
        joblib.dump(dict(schema='c3-hazard-v2-inclusive-research',features=FEATURES,model=model,plan=plan),artifact)
        loaded=load_research_model(artifact)
        np.testing.assert_array_equal(model.predict(test,calibrated=True),loaded.predict(test,calibrated=True))
        summary['saved_artifacts'].append(dict(path=artifact.name,bytes=artifact.stat().st_size,sha256=sha256(artifact)))
    (run_dir/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    provenance={str(p.relative_to(RESEARCH)):sha256(p) for p in [HERE/'hazard_v2.py',Path(__file__),
        RESEARCH/'scripts/build_and_evaluate.py',RESEARCH/'scripts/model_inference.py',
        RESEARCH/'experimental_multihorizon/hazard_model.py',RESEARCH/'experimental_multihorizon/train_evaluate.py',
        RESEARCH/'experimental_algorithm_comparison/models.py']}
    (run_dir/'source_hashes.json').write_text(json.dumps(provenance,indent=2))
    plot(run_dir,metrics,pd.DataFrame(bins),pd.concat(details))
    print(metrics[metrics.split.eq('test')][['method','horizon_days','brier','average_precision','precision_at_10','recall_at_10','ece_10_equal_width']].to_string(index=False),flush=True)


def plot(folder, metrics, bins, details):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for kind in ['calibration','precision_recall']:
        fig,axes=plt.subplots(1,3,figsize=(15,4))
        for h,ax in zip(HORIZONS,axes):
            for method in ['independent_sigmoid','lr_sigmoid','hgb_sigmoid']:
                if kind=='calibration':
                    b=bins[(bins.split=='test') & (bins.horizon_days==h) & (bins.method==method) & (bins['count']>0)]
                    ax.plot(b.mean_probability,b.observed_rate,'o-',label=method)
                else:
                    b=details[(details.horizon==h)&(details.method==method)]
                    precision,recall,_=precision_recall_curve(b.y,b.p); ax.plot(recall,precision,label=method)
            if kind=='calibration': ax.plot([0,1],[0,1],'k--'); ax.set(xlabel='Mean probability',ylabel='Observed rate')
            else: ax.set(xlabel='Recall',ylabel='Precision')
            ax.set(title=f'{h} days',xlim=(0,1),ylim=(0,1)); ax.legend(fontsize=7)
        fig.suptitle('Exploratory synthetic Azure benchmark; inclusive upper event bounds')
        fig.tight_layout(); fig.savefig(folder/f'{kind}.png',dpi=150); plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,default=HERE/'run_001')
    args=parser.parse_args(); run(args.data_dir,args.run_dir)
