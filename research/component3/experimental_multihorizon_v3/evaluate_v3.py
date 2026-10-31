"""Reproduce v2 in memory, then evaluate prespecified v3 methods in a NEW folder."""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[name]='1'
import sys
sys.dont_write_bytecode=True
import argparse,json,time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from methods_v3 import *
from train_evaluate_v2 import metric, GRID, development_folds, HazardV2, load_research_model
from train_evaluate import feasibility, load_csv, ordering, sha256

HERE=Path(__file__).resolve().parent
RESEARCH=HERE.parent
V2=RESEARCH/'experimental_multihorizon_v2/run_003'


def rank_changes(frame,base,p,method,h):
    rows=[]
    for date,g in frame.assign(base=base,p=p).groupby('prediction_time'):
        a=g.sort_values(['base','machineID'],ascending=[False,True]).machineID.tolist()
        b=g.sort_values(['p','machineID'],ascending=[False,True]).machineID.tolist()
        ar={mid:i+1 for i,mid in enumerate(a)}; br={mid:i+1 for i,mid in enumerate(b)}
        shifts=np.array([br[mid]-ar[mid] for mid in a])
        rows.append(dict(method=method,horizon=h,prediction_time=str(date),
            changed_rank_positions=int((shifts!=0).sum()),total_absolute_rank_shift=int(abs(shifts).sum()),
            max_absolute_rank_shift=int(abs(shifts).max()),top10_replacements=len(set(a[:10])-set(b[:10])),
            top10_order_positions_changed=sum(x!=y for x,y in zip(a[:10],b[:10]))))
    return rows


def run(data_dir,folder):
    require(folder.resolve().is_relative_to(HERE) and not folder.exists(),'Use a NEW directory inside v3')
    folder.mkdir(parents=True)
    plan=dict(seed=42,intervals=['(0,7]','(7,14]','(14,30]'],lr='fixed C=1, original baseline',
        hgb_grid=GRID,selection='TRAIN-only 3 expanding date folds, 30-day purge; mean RAW 3-horizon Brier',
        calibration='separate frozen sigmoid per interval; later eligible calibration risk sets only',
        projection='per-case score-only Euclidean projections; no outcome-fitted mapping',
        cohort=[1800,500,1300],followup_end_exclusive='2016-01-01',
        limitation='Repeatedly inspected synthetic Azure benchmark test period; exploratory, not factory validation')
    (folder/'plan.json').write_text(json.dumps(plan,indent=2))
    protected=sha256(RESEARCH/'models/c3_7day_sigmoid.joblib')
    sources=tuple(load_csv(data_dir,n,n!='machines') for n in ['machines','maint','failures','errors'])
    splits,_,_,counts=feasibility(sources,'2016-01-01')
    pp={}
    for s,frame in splits.items():
        out=outcomes(frame,sources[2],'2016-01-01')
        require(out.complete_30d.all(),'Incomplete full-cohort outcome follow-up')
        for h in HORIZONS: frame[f'label_failure_{h}d']=out[f'label_failure_{h}d'].to_numpy(dtype=int)
        pp[s]=periods(out)
    train,val,test=[splits[s] for s in ['train','validation','test']]
    targets={s:f[[f'label_failure_{h}d' for h in HORIZONS]].to_numpy() for s,f in splits.items()}
    saved=pd.read_csv(V2/'heldout_predictions.csv',parse_dates=['prediction_time'],float_precision='round_trip')
    keys=pd.MultiIndex.from_frame(test[['machineID','prediction_time']])
    def old_prob(method):
        data=saved[saved.method.eq(method)]
        require(not data.duplicated(['machineID','prediction_time','horizon']).any(),'Duplicate saved keys')
        wide=data.pivot(index=['machineID','prediction_time'],columns='horizon',values='p')
        truth=data.pivot(index=['machineID','prediction_time'],columns='horizon',values='y')
        require(set(wide.index)==set(keys),'Saved v2 cohort mismatch')
        np.testing.assert_array_equal(truth.reindex(keys)[list(HORIZONS)],targets['test'])
        return wide.reindex(keys)[list(HORIZONS)].to_numpy()
    predictions={s:{} for s in splits}; fitted_refs=[]; fidelity=[]
    for h in HORIZONS:
        label=f'label_failure_{h}d'; base=make_baseline().fit(train[FEATURES],train[label])
        model=CalibratedClassifierCV(FrozenEstimator(base),method='sigmoid').fit(val[FEATURES],val[label])
        fitted_refs.append(model)
    for s,frame in splits.items():
        ref=np.column_stack([m.predict_proba(frame[FEATURES])[:,1] for m in fitted_refs])
        predictions[s].update(independent_lr=ref,standard_pav=pav(ref),anchored_pav=anchored_pav(ref))
    delta=float(abs(predictions['test']['independent_lr']-old_prob('independent_sigmoid')).max())
    require(delta<1e-12,'Independent LR reproduction failed'); fidelity.append(dict(method='independent_lr',max_difference=delta))
    summary_v2=json.loads((V2/'summary.json').read_text())
    development=[]
    for candidate,params in enumerate(GRID):
        for a,b,info in development_folds(train):
            fitting,scoring=train.iloc[a],train.iloc[b]
            risk=periods(outcomes(fitting,sources[2],'2016-01-01'))
            # Reproduce v2 development and select v3 HGB on identical training folds.
            for kind in ['pooled','unpooled']:
                model=(HazardV2('hgb',params) if kind=='pooled' else UnpooledHazard('hgb',params)).fit(fitting,risk)
                p=model.predict(scoring,**({'calibrated':False} if kind=='unpooled' else {}))
                for j,h in enumerate(HORIZONS): development.append(dict(kind=kind,candidate=candidate,
                    fold=info['fold'],horizon=h,fitting_cases=len(a),scoring_cases=len(b),
                    brier=float(((scoring[f'label_failure_{h}d'].to_numpy()-p[:,j])**2).mean())))
    dev=pd.DataFrame(development)
    old_dev=pd.read_csv(V2/'development.csv')
    sort=['candidate','fold','horizon']
    np.testing.assert_allclose(dev[dev.kind.eq('pooled')].sort_values(sort).brier,
                               old_dev.sort_values(sort).brier,atol=1e-12,rtol=0)
    selected=int(dev[dev.kind.eq('unpooled')].groupby('candidate').brier.mean().sort_values(kind='stable').index[0])
    dev.to_csv(folder/'development.csv',index=False)
    runtimes=[]
    for kind in ['lr','hgb']:
        start=time.perf_counter()
        pooled=HazardV2(kind,summary_v2['selected_hgb_parameters'] if kind=='hgb' else None).fit(train,pp['train'])
        pooled.calibrate(val,pp['validation'])
        artifact=load_research_model(V2/f'hazard_v2_{kind}_research.joblib')
        p=pooled.predict(test,calibrated=True)
        np.testing.assert_allclose(p,artifact.predict(test,calibrated=True),atol=1e-12,rtol=0)
        delta=float(abs(p-old_prob(kind+'_sigmoid')).max())
        require(delta<1e-12,'Pooled v2 reproduction failed')
        fidelity.append(dict(method='pooled_'+kind,max_difference=delta))
        for s,frame in splits.items(): predictions[s]['pooled_'+kind]=pooled.predict(frame,calibrated=True)
        model=UnpooledHazard(kind,GRID[selected] if kind=='hgb' else None).fit(train,pp['train'])
        before=joblib.hash(model.models_); model.calibrate(val,pp['validation'])
        require(before==joblib.hash(model.models_),'Calibration modified fitted base estimators')
        for s,frame in splits.items():
            predictions[s]['unpooled_'+kind]=model.predict(frame)
            predictions[s]['unpooled_'+kind+'_raw']=model.predict(frame,calibrated=False)
        if kind=='lr':
            for s in splits: np.testing.assert_array_equal(predictions[s]['unpooled_lr'][:,0],predictions[s]['independent_lr'][:,0])
        path=folder/f'unpooled_{kind}_research.joblib'
        joblib.dump(dict(schema='unpooled-inclusive-v3',features=FEATURES,model=model),path)
        loaded=joblib.load(path)['model']
        np.testing.assert_array_equal(loaded.predict(test),model.predict(test))
        runtimes.append(dict(method='unpooled_'+kind,fit_verify_seconds=time.perf_counter()-start,artifact_bytes=path.stat().st_size))
    metrics=[]; bins=[]; ranking=[]; shifts=[]; details=[]; adjustments=[]; order={}
    for s,frame in splits.items():
        ref=predictions[s]['independent_lr']; truth=targets[s]
        for name,p in predictions[s].items():
            order[s+'/'+name]=ordering(p)
            if name!='independent_lr': require(order[s+'/'+name]['either']==0,'Ordering failed')
            if name=='standard_pav':
                require((((p-truth)**2).sum(axis=1)<=((ref-truth)**2).sum(axis=1)+1e-14).all(),'PAV loss guarantee failed')
            for j,h in enumerate(HORIZONS):
                row,bb,rr=metric(frame,truth[:,j],p[:,j],name,h)
                metrics.append(dict(split=s,**row)); bins.extend([dict(split=s,**b) for b in bb])
                if s=='test':
                    ranking.extend(rr); shifts.extend(rank_changes(frame,ref[:,j],p[:,j],name,h))
                    d=frame[['machineID','prediction_time']].copy();d['method']=name;d['horizon']=h;d['y']=truth[:,j];d['p']=p[:,j]
                    details.append(d)
                    change=p[:,j]-ref[:,j]; changed=change!=0
                    adjustments.append(dict(method=name,horizon=h,changed=int(changed.sum()),
                        changed_pct=100*changed.mean(),mean_absolute=float(abs(change).mean()),
                        max_absolute=float(abs(change).max()),increases=int((change>0).sum()),decreases=int((change<0).sum())))
    pd.DataFrame(metrics).to_csv(folder/'metrics_all_splits.csv',index=False,float_format='%.17g')
    comparison=pd.DataFrame(metrics);comparison=comparison[comparison.split.eq('test')]
    comparison.to_csv(folder/'comparison.csv',index=False,float_format='%.17g')
    pd.DataFrame(bins).to_csv(folder/'calibration_bins.csv',index=False)
    pd.DataFrame(ranking).to_csv(folder/'top10_by_date.csv',index=False)
    pd.DataFrame(shifts).to_csv(folder/'ranking_changes_by_date.csv',index=False)
    pd.DataFrame(shifts).groupby(['method','horizon']).agg(
        changed_rank_positions=('changed_rank_positions','sum'),total_absolute_rank_shift=('total_absolute_rank_shift','sum'),
        top10_replacements=('top10_replacements','sum'),top10_order_positions_changed=('top10_order_positions_changed','sum')).to_csv(folder/'ranking_changes_total.csv')
    pd.DataFrame(adjustments).to_csv(folder/'adjustments.csv',index=False)
    pd.concat(details).to_csv(folder/'heldout_predictions.csv',index=False,float_format='%.17g')
    # Fixed-prediction paired bootstrap; independent machine and date cluster views.
    intervals=[]
    for cluster in ['machineID','prediction_time']:
        labels=test[cluster].to_numpy();groups=np.unique(labels);rng=np.random.default_rng(42)
        draws=rng.integers(0,len(groups),(1000,len(groups)))
        ref=(targets['test']-predictions['test']['independent_lr'])**2
        for name in ['standard_pav','anchored_pav','unpooled_lr','unpooled_hgb']:
            diff=(targets['test']-predictions['test'][name])**2-ref
            means=np.array([diff[labels==g].mean(axis=0) for g in groups]); boot=means[draws].mean(axis=1)
            for j,h in enumerate(HORIZONS):intervals.append(dict(cluster=cluster,method=name,horizon=h,delta=diff[:,j].mean(),
                low=np.quantile(boot[:,j],.025),high=np.quantile(boot[:,j],.975)))
    pd.DataFrame(intervals).to_csv(folder/'paired_cluster_deltas.csv',index=False)
    require(sha256(RESEARCH/'models/c3_7day_sigmoid.joblib')==protected,'Deployed artifact changed')
    risksets={s:[dict(interval=int(i),positives=int(g.event.sum()),negatives=int((1-g.event).sum()),cases=len(g))
        for i,g in risk.groupby('interval')] for s,risk in pp.items()}
    summary=dict(schema='c3-v3-research',plan=plan,risk_sets=risksets,v2_reproduction=fidelity,ordering=order,
        selected_unpooled_hgb=GRID[selected],versions=dict(sklearn=sklearn.__version__,numpy=np.__version__,pandas=pd.__version__),
        deployed_sha256=protected,split_counts={s:len(f) for s,f in splits.items()},runtime=runtimes,
        total_loss_pav_before=float(((predictions['test']['independent_lr']-targets['test'])**2).sum()),
        total_loss_pav_after=float(((predictions['test']['standard_pav']-targets['test'])**2).sum()))
    (folder/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    provenance={str(p.relative_to(RESEARCH)):sha256(p) for p in [Path(__file__),HERE/'methods_v3.py',
        RESEARCH/'experimental_multihorizon_v2/hazard_v2.py',RESEARCH/'experimental_multihorizon_v2/train_evaluate_v2.py',
        V2/'heldout_predictions.csv',V2/'comparison.csv',V2/'development.csv',V2/'summary.json']}
    for name in ['machines','maint','failures','errors']:provenance['raw/PdM_'+name+'.csv']=sha256(data_dir/f'PdM_{name}.csv')
    (folder/'source_hashes.json').write_text(json.dumps(provenance,indent=2))
    plot(folder,pd.DataFrame(bins),pd.concat(details))
    print(comparison[['method','horizon_days','brier','average_precision','precision_at_10','recall_at_10','ece_10_equal_width']].to_string(index=False),flush=True)


def plot(folder,bins,details):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=['independent_lr','standard_pav','anchored_pav','unpooled_lr','unpooled_hgb']
    for kind in ['calibration','precision_recall']:
        fig,axes=plt.subplots(1,3,figsize=(16,4.5))
        for h,ax in zip(HORIZONS,axes):
            for name in names:
                if kind=='calibration':
                    d=bins[(bins.split=='test')&(bins.horizon_days==h)&(bins.method==name)&(bins['count']>0)]
                    ax.plot(d.mean_probability,d.observed_rate,'o-',label=name)
                else:
                    from sklearn.metrics import precision_recall_curve
                    d=details[(details.horizon==h)&(details.method==name)]
                    precision,recall,_=precision_recall_curve(d.y,d.p);ax.plot(recall,precision,label=name)
            if kind=='calibration':ax.plot([0,1],[0,1],'k--');ax.set(xlabel='Mean probability',ylabel='Observed rate')
            else:ax.set(xlabel='Recall',ylabel='Precision')
            ax.set(title=f'{h} days',xlim=(0,1),ylim=(0,1));ax.legend(fontsize=7)
        fig.suptitle('Exploratory synthetic Azure benchmark: inclusive horizons, reused test set')
        fig.tight_layout();fig.savefig(folder/f'{kind}.png',dpi=150);plt.close(fig)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,default=HERE/'run_001');args=parser.parse_args();run(args.data_dir,args.run_dir)
