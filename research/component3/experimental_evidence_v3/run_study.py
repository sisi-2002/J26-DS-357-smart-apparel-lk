"""Execute a prespecified frozen-v3 study in a NEW local output directory."""
import os
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '1'
import sys
sys.dont_write_bytecode = True
import argparse
from contextlib import contextmanager
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from evidence import (ROOT, FEATURES, HORIZONS, KEYS, EVIDENCE, original, inference, outcomes,
                      UnpooledHazard, conditions, reconstruct, require, validate_probabilities,
                      verify_splits, secondary_feasibility)
from evaluation import metrics, cluster_weights, bootstrap_metrics, intervals

HERE = Path(__file__).resolve().parent
ARTIFACT = ROOT / 'research/component3/experimental_multihorizon_v3/run_003/unpooled_hgb_research.joblib'


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def forbid_fitting():
    def forbidden(*args, **kwargs):
        raise RuntimeError('No estimator or preprocessing fitting permitted in the frozen evidence study')
    saved = []
    for cls in [Pipeline, ColumnTransformer, CalibratedClassifierCV, HistGradientBoostingClassifier, LogisticRegression, SimpleImputer]:
        for method in ['fit', 'fit_transform']:
            if hasattr(cls, method):
                saved.append((cls, method, getattr(cls, method)))
                setattr(cls, method, forbidden)
    try:
        yield
    finally:
        for cls, name, value in saved:
            setattr(cls, name, value)


def load_inputs(data_dir):
    manifest_path = ROOT / 'backend/app/components/component3/experimental/v3_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    require(importlib.metadata.version('scikit-learn') == manifest['sklearn'], 'Audited sklearn version required')
    hashes = {str(manifest_path): digest(manifest_path)}
    deployed=ROOT/'research/component3/models/c3_7day_sigmoid.joblib'
    require(digest(deployed)=='20908cc0c0466ddc5563d542ff2068223cddd0ed2f6cded620a65f9fbc7c3bcc','Deployed LR artifact identity mismatch')
    hashes[str(deployed)]=digest(deployed)
    for p, expected in [(ROOT / p, h) for p, h in manifest['source_sha256'].items()] + [(ARTIFACT, manifest['artifact_sha256'])] + [(data_dir / n, h) for n, h in manifest['data_sha256'].items()]:
        require(p.is_file() and digest(p) == expected, f'Missing or mismatched audited input: {p}')
        hashes[str(p)] = expected
    sources = inference.read_demo_sources(data_dir)
    audit = []
    for name, frame in zip(['machines', 'maintenance', 'failures', 'errors'], sources):
        require(not frame.isna().any().any(), 'Missing raw fields')
        require(frame.machineID.isin(range(1, 101)).all(), 'Unknown machine IDs')
        audit.append(dict(source=name, rows=len(frame), columns=list(frame), duplicates=int(frame.duplicated().sum()),
                          categories=sorted(frame.iloc[:,-1].astype(str).unique().tolist()) if name != 'machines' else None))
    require(sources[0].machineID.is_unique and len(sources[0]) == 100, 'Expected 100 unique machines')
    bundle = joblib.load(ARTIFACT)
    require(bundle['schema'] == 'unpooled-inclusive-v3' and bundle['features'] == FEATURES, 'Artifact schema mismatch')
    model = bundle['model']
    require(isinstance(model, UnpooledHazard) and model.algorithm == 'hgb', 'Wrong v3 variant')
    require(len(model.models_) == len(model.calibrators_) == 3, 'Missing interval estimators')
    for base, calibrated in zip(model.models_, model.calibrators_):
        require(calibrated.method == 'sigmoid' and calibrated.estimator.estimator is base, 'Calibration identity mismatch')
        require(list(base.feature_names_in_) == FEATURES, 'Preprocessing schema mismatch')
    return model, sources, hashes, audit


def describe(frame, condition):
    rows=[]
    for name, col in EVIDENCE.items():
        value=frame[col]; valid=value.dropna()
        rows.append(dict(condition=condition, evidence=name, field=col, cases=len(value), available=int(value.notna().sum()),
                         missing=int(value.isna().sum()), zeros=int(value.eq(0).sum()), distinct=int(valid.nunique()),
                         mean=valid.mean(), minimum=valid.min(), q25=valid.quantile(.25), median=valid.median(), q75=valid.quantile(.75), maximum=valid.max()))
    return rows


def associations(frame, y, p):
    groups=[]; correlations=[]
    for name, col in EVIDENCE.items():
        values=frame[col]
        bins=pd.qcut(values,5,duplicates='drop') if values.nunique()>1 else pd.Series('constant',index=frame.index)
        bins=bins.astype(object);bins.loc[values.isna()]='missing'
        for j,h in enumerate(HORIZONS):
            loss=(p[:,j]-y[:,j])**2
            for label in pd.unique(bins):
                mask=(bins==label).to_numpy(); observed=values[mask]
                groups.append(dict(evidence=name,horizon=h,group=str(label),cases=int(mask.sum()),
                                   machines=frame.loc[mask,'machineID'].nunique(),dates=frame.loc[mask,'prediction_time'].nunique(),
                                   evidence_min=observed.min(),evidence_max=observed.max(),positive_cases=int(y[mask,j].sum()),
                                   prevalence=y[mask,j].mean(),mean_probability=p[mask,j].mean(),mean_squared_loss=loss[mask].mean()))
            subsets=[('overall','all',np.ones(len(frame),dtype=bool))]
            for group_col in ['prediction_time','machine_model']:
                subsets += [(group_col,str(key),(frame[group_col]==key).to_numpy()) for key in frame[group_col].unique()]
            for kind,key,mask in subsets:
                v=values[mask]; l=pd.Series(loss[mask],index=v.index)
                correlations.append(dict(evidence=name,horizon=h,scope=kind,group=key,cases=int(mask.sum()),
                                         available=int(v.notna().sum()),rho=v.corr(l,method='spearman') if v.nunique()>1 and l.nunique()>1 else np.nan))
    return groups,correlations


def run(data_dir, out, bootstrap_draws=1000):
    require(out.resolve().is_relative_to(HERE) and not out.exists(), 'Use a NEW run directory inside experimental_evidence_v3')
    specs=conditions()
    plan=dict(schema='c3-evidence-v3-frozen-1', conditions=specs, horizons=HORIZONS, bootstrap_draws=bootstrap_draws,
              bootstrap_seeds=dict(machineID=42,prediction_time=43), outcome='realized squared probability error',
              selection='No outcome-driven feature/condition/model/threshold selection; descriptive five-quantile bins with ties preserved',
              followup_end_exclusive='2016-01-01', secondary='Feasibility audit only unless independent chronological cases exist',
              limitation='Synthetic Azure; repeatedly inspected test period; exploratory; no per-machine reliability label')
    out.mkdir(parents=True);(out/'plan.json').write_text(json.dumps(plan,indent=2))
    started=time.perf_counter()
    model,sources,hashes,audit=load_inputs(data_dir)
    model_before=joblib.hash(model)
    panel=original.build_panel(*sources)
    old=ROOT/'research/component3/outputs_corrected/machine_prediction_examples.csv'
    stored=pd.read_csv(old,parse_dates=['prediction_time'])
    pd.testing.assert_frame_equal(panel.set_index(KEYS).sort_index(),stored.set_index(KEYS).sort_index(),check_dtype=False,check_exact=False,atol=1e-12,rtol=1e-12)
    hashes[str(old)]=digest(old)
    splits=verify_splits(panel)
    followup=[]
    for name,f in splits.items():
        o=outcomes(f,sources[2],'2016-01-01')
        require(o.complete_30d.all(),'Incomplete full-cohort follow-up: stop; censored negatives must not be substituted')
        for h in HORIZONS:f[f'label_failure_{h}d']=o[f'label_failure_{h}d'].to_numpy(dtype=int)
        followup.append(dict(split=name,cases=len(f),dates=f.prediction_time.nunique(),first=str(f.prediction_time.min()),last=str(f.prediction_time.max()),
                             minimum_followup_days=float(o.followup_days.min()),incomplete_30d=int((~o.complete_30d).sum()),
                             positives={h:int(f[f'label_failure_{h}d'].sum()) for h in HORIZONS}))
    test=splits['test'].reset_index(drop=True);y=test[[f'label_failure_{h}d' for h in HORIZONS]].to_numpy()
    require((np.diff(y,axis=1)>=0).all(),'Non-nested labels')
    reference_features=reconstruct(sources,test,specs[0])
    pd.testing.assert_frame_equal(reference_features[FEATURES],test[FEATURES],check_dtype=False,check_exact=True)
    reference=validate_probabilities(model.predict(reference_features),len(test))
    saved_path=ARTIFACT.parent/'heldout_predictions.csv'
    saved=pd.read_csv(saved_path,float_precision='round_trip',parse_dates=['prediction_time'])
    saved=saved[saved.method.eq('unpooled_hgb')]
    require(not saved.duplicated(KEYS+['horizon']).any(),'Duplicate saved prediction keys')
    keys=pd.MultiIndex.from_frame(test[KEYS])
    wide=saved.pivot(index=KEYS,columns='horizon',values='p');truth=saved.pivot(index=KEYS,columns='horizon',values='y')
    require(len(wide)==len(test) and set(wide.index)==set(keys),'Saved cohort mismatch')
    np.testing.assert_array_equal(reference,wide.reindex(keys)[list(HORIZONS)].to_numpy())
    np.testing.assert_array_equal(y,truth.reindex(keys)[list(HORIZONS)].to_numpy())
    hashes[str(saved_path)]=digest(saved_path)
    comparison_path=ARTIFACT.parent/'comparison.csv';hashes[str(comparison_path)]=digest(comparison_path)
    comparison=pd.read_csv(comparison_path,float_precision='round_trip')
    profile=pd.concat([test[KEYS+['machine_model','machine_age_years']],reference_features.drop(columns=['machine_model','machine_age_years'])],axis=1)
    for j,h in enumerate(HORIZONS):
        profile[f'p{h}']=reference[:,j];profile[f'y{h}']=y[:,j];profile[f'loss{h}']=(reference[:,j]-y[:,j])**2
    # Categories are descriptive only and are never added to the fitted feature schema.
    for category in sorted(sources[3].errorID.unique()):
        profile[f'{category}_prior']=[int(((sources[3].machineID==mid)&(sources[3].datetime<t)&(sources[3].errorID==category)).sum()) for mid,t in test[KEYS].itertuples(index=False,name=None)]
    profile.to_csv(out/'baseline_evidence_profiles.csv',index=False,float_format='%.17g')
    groups,corr=associations(profile,y,reference)
    pd.DataFrame(groups).to_csv(out/'baseline_evidence_strata.csv',index=False)
    pd.DataFrame(corr).to_csv(out/'baseline_descriptive_correlations.csv',index=False)
    feasibility=secondary_feasibility(splits)
    require(feasibility['eligible_days']==0,'Independent secondary period detected: review protocol before proceeding')
    pd.DataFrame([dict(horizon=h,comparison='C0 risk-only vs C1 risk-plus-evidence',**feasibility) for h in HORIZONS]).to_csv(out/'secondary_feasibility.csv',index=False)
    imputation=[]
    for interval,base in enumerate(model.models_,1):
        numeric=base.named_steps['transform'].named_transformers_['num'].named_steps['impute']
        for i,col in enumerate(original.FEATURE_NUMERIC):
            imputation.append(dict(interval=interval,feature=col,training_median=float(numeric.statistics_[i]),
                                   training_missing_indicator=i in numeric.indicator_.features_))
    pd.DataFrame(imputation).to_csv(out/'frozen_imputation_audit.csv',index=False)
    weight_sets={c:cluster_weights(test[c].to_numpy(),bootstrap_draws,seed) for c,seed in [('machineID',42),('prediction_time',43)]}
    boot_reference={}; base_metrics={}
    for j,h in enumerate(HORIZONS):
        row,_,selected=metrics(test,y[:,j],reference[:,j]);base_metrics[h]=row
        expected=comparison[(comparison.method=='unpooled_hgb')&(comparison.horizon_days==h)].iloc[0]
        for name in ['brier','average_precision','roc_auc','precision_at_10','recall_at_10','ece_10_equal_width']:
            require(abs(row[name]-expected[name])<1e-12,f'Reference metric parity failed: {h}/{name}')
        for cluster,w in weight_sets.items():boot_reference[(cluster,h)]=bootstrap_metrics(y[:,j],reference[:,j],selected,w)
    metric_rows=[];calibration=[];top_rows=[];uncertainty=[];descriptive=[];changes=[]
    for number,spec in enumerate(specs):
        condition=spec['condition'];begin=time.perf_counter()
        features=reference_features if condition=='full' else reconstruct(sources,test,spec)
        require(features.index.equals(test.index),'Changed cohort alignment')
        p=reference if condition=='full' else validate_probabilities(model.predict(features),len(test))
        details=test[KEYS].copy()
        for j,h in enumerate(HORIZONS):details[f'y{h}']=y[:,j];details[f'p{h}']=p[:,j];details[f'loss{h}']=(p[:,j]-y[:,j])**2
        for col in [*EVIDENCE.values(),'available_maintenance_records','available_failure_records','available_error_records']:
            details[col]=features[col]
        details.to_csv(out/f'predictions_{condition}.csv',index=False,float_format='%.17g')
        descriptive.extend(describe(features,condition))
        for j,h in enumerate(HORIZONS):
            row,bins,selected=metrics(test,y[:,j],p[:,j]);delta=p[:,j]-reference[:,j]
            train_prevalence=splits['train'][f'label_failure_{h}d'].mean()
            metric_rows.append(dict(condition=condition,family=spec['family'],removal=spec['removal'],seed=spec['seed'],horizon=h,
                                    ordering_violations=int((np.diff(p,axis=1)<0).any(axis=1).sum()),
                                    train_prevalence_baseline_brier=float(np.mean((y[:,j]-train_prevalence)**2)),
                                    test_prevalence_descriptive_brier=float(y[:,j].mean()*(1-y[:,j].mean())),
                                    **row,**{f'delta_{k}':row[k]-base_metrics[h][k] for k in ['brier','average_precision','roc_auc','precision_at_10','recall_at_10']},
                                    changed_probabilities=int((delta!=0).sum()),mean_absolute_probability_change=float(abs(delta).mean()),
                                    max_absolute_probability_change=float(abs(delta).max()),mean_probability_change=float(delta.mean())))
            calibration += [dict(condition=condition,horizon=h,**b) for b in bins]
            for date,g in test.assign(y=y[:,j],selected=selected,p=p[:,j]).groupby('prediction_time'):
                ranked=g[g.selected].sort_values(['p','machineID'],ascending=[False,True])
                top_rows.append(dict(condition=condition,horizon=h,prediction_time=str(date),cases=len(g),positive_cases=int(g.y.sum()),
                                     captured=int(ranked.y.sum()),precision_at_10=float(ranked.y.mean()),recall_at_10=ranked.y.sum()/g.y.sum() if g.y.sum() else np.nan,
                                     machine_ids=';'.join(map(str,ranked.machineID.tolist()))))
            for cluster,w in weight_sets.items():
                current=bootstrap_metrics(y[:,j],p[:,j],selected,w)
                for ci in intervals(current,boot_reference[(cluster,h)]):
                    uncertainty.append(dict(condition=condition,horizon=h,cluster=cluster,clusters=test[cluster].nunique(),
                                            delta=row[ci['metric']]-base_metrics[h][ci['metric']],**ci))
        changes.append(dict(condition=condition,seconds=time.perf_counter()-begin,masked_features=';'.join(spec['mask']),
                            maintenance_events_removed=int((reference_features.available_maintenance_records-features.available_maintenance_records).sum()),
                            error_events_removed=int((reference_features.available_error_records-features.available_error_records).sum()),
                            zero_prior_maintenance=int(features.available_maintenance_records.eq(0).sum()),
                            zero_prior_errors=int(features.available_error_records.eq(0).sum())))
        print(f'{number+1}/{len(specs)} {condition}: '+', '.join(f'{h}d={metric_rows[-3+i]["brier"]:.6f}' for i,h in enumerate(HORIZONS)),flush=True)
    for name,rows in [('metrics',metric_rows),('calibration_bins',calibration),('top10_by_date',top_rows),
                      ('paired_cluster_intervals',uncertainty),('evidence_distributions',descriptive),('condition_audit',changes)]:
        pd.DataFrame(rows).to_csv(out/f'{name}.csv',index=False,float_format='%.17g')
    thinning=pd.DataFrame(metric_rows);thinning=thinning[thinning.family.eq('thinning')]
    thinning['stream']=thinning.condition.str.split('_drop').str[0]
    thinning.groupby(['stream','removal','horizon']).agg(replicates=('brier','size'),mean_brier=('brier','mean'),min_brier=('brier','min'),max_brier=('brier','max'),
        mean_ap=('average_precision','mean'),min_ap=('average_precision','min'),max_ap=('average_precision','max'),mean_precision10=('precision_at_10','mean'),mean_recall10=('recall_at_10','mean')).reset_index().to_csv(out/'thinning_replicate_summary.csv',index=False)
    require(joblib.hash(model)==model_before,'Frozen model state changed')
    for name,h in hashes.items():require(digest(name)==h,'Audited input changed during execution')
    summary=dict(schema=plan['schema'],run_seconds=time.perf_counter()-started,conditions=len(specs),cohort_cases=len(test),
                 exact_reference_probability_values=reference.size,ordering_violations_total=sum(r['ordering_violations'] for r in metric_rows),
                 split_audit=followup,secondary=feasibility,raw_audit=audit,artifact_sha256=digest(ARTIFACT),
                 fitted_state_before=model_before,fitted_state_after=joblib.hash(model),source_sha256=hashes,
                 versions={name:importlib.metadata.version(name) for name in ['numpy','pandas','scikit-learn','scipy','joblib','matplotlib']})
    for name in ['evidence.py','evaluation.py','run_study.py']:
        summary['source_sha256'][str(HERE/name)]=digest(HERE/name)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--bootstrap-draws',type=int,default=1000)
    args=parser.parse_args();require(args.bootstrap_draws>=100,'Use at least 100 bootstrap draws')
    with forbid_fitting():run(args.data_dir,args.out,args.bootstrap_draws)
