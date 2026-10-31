import os
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score
from evidence import (conditions, historical_sources, stable_uniform, reconstruct, outcomes, verify_splits,
                      secondary_feasibility, validate_probabilities, FEATURES, MAINTENANCE, ERRORS, ROOT)
from evaluation import top_membership, metrics, weighted_ap, cluster_weights, bootstrap_metrics, intervals
from run_study import load_inputs, forbid_fitting, digest

T=pd.Timestamp('2015-09-02 06:00:00')


def sources():
    machines=pd.DataFrame(dict(machineID=[1],model=['model1'],age=[5]))
    times=[T-pd.Timedelta(days=90),T-pd.Timedelta(days=7),T,T+pd.Timedelta(days=1)]
    def events(category,value):
        return pd.DataFrame({'machineID':[1]*4,'datetime':times,category:[value]*4})
    return machines,events('comp','comp1'),events('failure','comp1'),events('errorID','error1')


def landmarks():return pd.DataFrame({'machineID':[1],'prediction_time':[T]})


def test_feature_boundaries_strict_past_and_missing_recency():
    full=conditions()[0];x=reconstruct(sources(),landmarks(),full).iloc[0]
    assert x.maintenance_events_prior==2 and x.maintenance_events_7d==1 and x.maintenance_events_90d==2
    assert x.days_since_last_maintenance==7 and x.observed_maintenance_span_days==90
    empty=tuple([sources()[0]]+[f[f.datetime>=T] for f in sources()[1:]])
    z=reconstruct(empty,landmarks(),full).iloc[0]
    assert z.maintenance_events_prior==0 and np.isnan(z.days_since_last_maintenance)
    assert z.published_field_completeness_pct==100


def test_truncation_includes_lower_boundary_without_modifying_truth_source():
    source=sources();before=[f.copy(deep=True) for f in source]
    spec=dict(conditions()[0],streams=[1],days=7)
    x=reconstruct(source,landmarks(),spec).iloc[0]
    assert x.maintenance_events_prior==1 and x.failure_events_prior==2 and x.error_events_prior==2
    for a,b in zip(before,source):pd.testing.assert_frame_equal(a,b)


def test_thinning_keys_are_persistent_reproducible_and_future_independent():
    source=sources()[1];a=stable_uniform(source,11)
    pd.testing.assert_series_equal(a,stable_uniform(source.copy(),11))
    assert not np.array_equal(a,stable_uniform(source,29))
    appended=pd.concat([source,pd.DataFrame({'machineID':[1],'datetime':[T+pd.Timedelta(days=100)],'comp':['comp1']})],ignore_index=True)
    np.testing.assert_array_equal(a,stable_uniform(appended,11).iloc[:len(a)])
    assert set(source.index[a>=.75]).issubset(set(source.index[a>=.5]))
    spec=dict(conditions()[0],streams=[1],removal=.5,seed=11)
    past=historical_sources(sources(),T,spec)
    assert (past[1].datetime<T).all() and (past[2].datetime<T).all()


def test_thinning_can_remove_all_records_without_inventing_recency():
    spec=dict(conditions()[0],streams=[1,3],removal=1.,seed=11)
    x=reconstruct(sources(),landmarks(),spec).iloc[0]
    assert x.maintenance_events_prior==x.error_events_prior==0
    assert np.isnan(x.days_since_last_error) and np.isnan(x.observed_maintenance_span_days)


def test_field_mask_uses_nan_and_never_changes_categories_or_completeness():
    spec=next(s for s in conditions() if s['condition']=='mask_maintenance')
    x=reconstruct(sources(),landmarks(),spec)
    assert x[MAINTENANCE].isna().all().all()
    assert not x[ERRORS].isna().all().all()
    assert x.machine_model.iloc[0]=='model1' and x.available_maintenance_records.iloc[0]==2
    assert x.published_field_completeness_pct.iloc[0]==100
    with pytest.raises(ValueError,match='Only fitted numeric'):
        reconstruct(sources(),landmarks(),dict(spec,mask=['machine_model']))


@pytest.mark.parametrize('delay,expected',[(0,[0,0,0]),(7,[1,1,1]),(14,[0,1,1]),(30,[0,0,1]),(31,[0,0,0])])
def test_inclusive_event_boundaries(delay,expected):
    f=pd.DataFrame({'machineID':[1,1],'datetime':[T,T+pd.Timedelta(days=delay)]})
    o=outcomes(landmarks(),f,T+pd.Timedelta(days=40))
    assert o[['label_failure_7d','label_failure_14d','label_failure_30d']].iloc[0].tolist()==expected


def test_right_censoring_never_turns_unknown_negative_into_zero():
    f=pd.DataFrame({'machineID':[1],'datetime':[T+pd.Timedelta(days=30)]})
    o=outcomes(landmarks(),f,T+pd.Timedelta(days=30))
    assert not o.complete_30d.iloc[0] and np.isnan(o.label_failure_30d.iloc[0])
    o=outcomes(landmarks(),f,T+pd.Timedelta(days=10))
    assert o.label_failure_7d.iloc[0]==0 and np.isnan(o.label_failure_14d.iloc[0])


def panel():
    records=[]
    for name,start,n in [('train','2015-02-01 06:00',18),('validation','2015-07-02 06:00',5),('test','2015-09-02 06:00',13)]:
        records.extend(dict(split=name,prediction_time=t,machineID=mid) for t in pd.date_range(start,periods=n,freq='7D') for mid in range(1,101))
    return pd.DataFrame(records)


def test_embargo_and_secondary_training_eligibility():
    p=panel();s=verify_splits(p);audit=secondary_feasibility(s)
    assert audit['eligible_days']==0 and audit['status']=='not_estimable_for_frozen_release'
    # No test labels enter secondary feasibility or any fitting.
    s['test']['label_failure_7d']=np.arange(1300)%2
    assert secondary_feasibility(s)==audit
    p.loc[p.split.eq('validation'),'prediction_time']=pd.Timestamp('2015-06-10 06:00')
    with pytest.raises(ValueError,match='embargo'):verify_splits(p)


@pytest.mark.parametrize('p',[[[.1,.2,.3]],[[0,0,0]],[[1,1,1]]])
def test_valid_ordering(p):np.testing.assert_array_equal(validate_probabilities(p,1),p)


@pytest.mark.parametrize('p',[[[.3,.2,.4]],[[np.nan,.2,.4]],[[-.1,.2,.4]],[[.1,.2,1.1]],[[.1,.2]]])
def test_invalid_ordering_or_shape(p):
    with pytest.raises(ValueError):validate_probabilities(p,1)


def test_grouped_top10_and_known_metrics():
    f=pd.DataFrame({'machineID':list(range(1,21))*2,'prediction_time':[T]*20+[T+pd.Timedelta(days=7)]*20})
    p=np.full(40,.5);y=np.tile([1]*10+[0]*10,2)
    row,_,selected=metrics(f,y,p)
    assert selected.sum()==20 and row['top10_captured']==20
    assert row['brier']==.25 and row['precision_at_10']==row['recall_at_10']==1
    assert row['ece_10_equal_width']==0 and row['average_precision']==.5
    with pytest.raises(ValueError):top_membership(pd.concat([f,f]),np.tile(p,2))


def test_weighted_bootstrap_ap_matches_sklearn_with_ties():
    y=np.array([1,0,0,1,1,0]);p=np.array([.1,.5,.5,.9,.9,.2])
    w=np.random.default_rng(42).integers(0,5,size=(30,6)).astype(float)
    actual=weighted_ap(y,p,w)
    expected=[average_precision_score(y,p,sample_weight=row) for row in w]
    np.testing.assert_allclose(actual,expected,atol=1e-15,rtol=0)


def test_cluster_weights_keep_whole_machine_trajectories_and_paired_zero_delta():
    labels=np.array([1,1,2,2,3,3]);w=cluster_weights(labels,200,42)
    assert np.array_equal(w[:,0],w[:,1]) and np.array_equal(w[:,4],w[:,5])
    assert np.array_equal(w,cluster_weights(labels,200,42)) and (w.sum(axis=1)==6).all()
    y=np.array([0,1,1,0,1,0]);p=np.linspace(.1,.6,6);sel=np.ones(6,dtype=bool)
    b=bootstrap_metrics(y,p,sel,w)
    for ci in intervals(b,b):assert ci['low']==ci['high']==0


def test_no_fitting_guard_restores_methods():
    from sklearn.ensemble import HistGradientBoostingClassifier
    before=HistGradientBoostingClassifier.fit
    with forbid_fitting():
        with pytest.raises(RuntimeError,match='No estimator'):HistGradientBoostingClassifier().fit([[0],[1]],[0,1])
    assert HistGradientBoostingClassifier.fit is before


def test_missing_or_inconsistent_inputs_fail_explicitly(tmp_path):
    with pytest.raises(ValueError,match='Missing or mismatched'):load_inputs(tmp_path)


def test_real_saved_model_cohort_and_state_preservation():
    data=os.environ.get('C3_EVIDENCE_DATA_DIR')
    if not data:pytest.skip('Set C3_EVIDENCE_DATA_DIR for actual-source verification')
    with forbid_fitting():
        model,sources,hashes,_=load_inputs(Path(data));before=joblib.hash(model)
        f=verify_splits(__import__('evidence').original.build_panel(*sources))['test'].reset_index(drop=True)
        full=reconstruct(sources,f,conditions()[0]);p=validate_probabilities(model.predict(full),1300)
        saved=pd.read_csv(ROOT/'research/component3/experimental_multihorizon_v3/run_003/heldout_predictions.csv',float_precision='round_trip',parse_dates=['prediction_time'])
        saved=saved[saved.method.eq('unpooled_hgb')].pivot(index=['machineID','prediction_time'],columns='horizon',values='p')
        np.testing.assert_array_equal(p,saved.reindex(pd.MultiIndex.from_frame(f[['machineID','prediction_time']]))[[7,14,30]].to_numpy())
        mask=next(s for s in conditions() if s['condition']=='mask_errors')
        degraded=reconstruct(sources,f.iloc[:100],mask)
        validate_probabilities(model.predict(degraded),100)
        assert list(model.models_[0].feature_names_in_)==FEATURES
        assert joblib.hash(model)==before
        for path,h in hashes.items():assert digest(path)==h


def test_completed_run_matched_cohorts_metrics_and_provenance():
    location=os.environ.get('C3_EVIDENCE_RUN_DIR')
    if not location:pytest.skip('Set C3_EVIDENCE_RUN_DIR to audit all saved conditions')
    import json
    from sklearn.metrics import brier_score_loss,roc_auc_score
    folder=Path(location);summary=json.loads((folder/'summary.json').read_text());plan=json.loads((folder/'plan.json').read_text())
    assert summary['conditions']==len(conditions())==40
    assert summary['fitted_state_before']==summary['fitted_state_after']
    assert summary['ordering_violations_total']==0
    for name,h in summary['source_sha256'].items():assert digest(name)==h
    baseline=pd.read_csv(folder/'predictions_full.csv',float_precision='round_trip')
    reported=pd.read_csv(folder/'metrics.csv',float_precision='round_trip')
    assert len(reported)==120
    for spec in conditions():
        condition=spec['condition'];f=pd.read_csv(folder/f'predictions_{condition}.csv',float_precision='round_trip')
        pd.testing.assert_frame_equal(f[['machineID','prediction_time','y7','y14','y30']],baseline[['machineID','prediction_time','y7','y14','y30']])
        validate_probabilities(f[['p7','p14','p30']].to_numpy(),1300)
        for h in [7,14,30]:
            row=reported[(reported.condition==condition)&(reported.horizon==h)].iloc[0]
            assert row.brier==pytest.approx(brier_score_loss(f[f'y{h}'],f[f'p{h}']),abs=1e-15)
            assert row.average_precision==pytest.approx(average_precision_score(f[f'y{h}'],f[f'p{h}']),abs=1e-15)
            assert row.roc_auc==pytest.approx(roc_auc_score(f[f'y{h}'],f[f'p{h}']),abs=1e-15)
            selected=f.sort_values(['prediction_time',f'p{h}','machineID'],ascending=[True,False,True]).groupby('prediction_time').head(10)
            assert len(selected)==130 and row.top10_captured==selected[f'y{h}'].sum()
            assert row.precision_at_10==pytest.approx(selected[f'y{h}'].sum()/130)
    ci=pd.read_csv(folder/'paired_cluster_intervals.csv')
    assert len(ci)==40*3*2*4 and ci.valid_draws.le(plan['bootstrap_draws']).all()
    assert (ci[ci.condition.eq('full')][['low','high']]==0).all().all()
    assert set(ci.clusters)=={100,13}
    thin=pd.read_csv(folder/'thinning_replicate_summary.csv');assert thin.replicates.eq(5).all() and len(thin)==18
    profile=pd.read_csv(folder/'baseline_evidence_profiles.csv');cats=[c for c in profile if c.startswith('error') and c.endswith('_prior') and c!='error_events_prior']
    np.testing.assert_array_equal(profile[cats].sum(axis=1),profile.error_events_prior)
    assert profile.published_field_completeness_pct.eq(100).all()
    secondary=pd.read_csv(folder/'secondary_feasibility.csv')
    assert len(secondary)==3 and secondary.eligible_days.eq(0).all()


def test_saved_perturbation_repeats_actual_predictions():
    location=os.environ.get('C3_EVIDENCE_RUN_DIR');data=os.environ.get('C3_EVIDENCE_DATA_DIR')
    if not location or not data:pytest.skip('Configure real inputs and completed run')
    saved=pd.read_csv(Path(location)/'predictions_maintenance_drop50_s11.csv',parse_dates=['prediction_time'],float_precision='round_trip').iloc[:100]
    spec=next(s for s in conditions() if s['condition']=='maintenance_drop50_s11')
    with forbid_fitting():
        model,source,_,_=load_inputs(Path(data))
        a=reconstruct(source,saved,spec);b=reconstruct(source,saved,spec)
        pd.testing.assert_frame_equal(a,b)
        np.testing.assert_array_equal(model.predict(a),saved[['p7','p14','p30']].to_numpy())
