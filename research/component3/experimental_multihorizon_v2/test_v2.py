import os
os.environ['OMP_NUM_THREADS']='1'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from hazard_v2 import HazardV2, FEATURES, outcomes, periods, cumulative_probabilities, load_research_model
from train_evaluate_v2 import development_folds, sha256
from model_inference import machine_features

T=pd.Timestamp('2015-09-02 06:00')

def landmark(event_days=None, followup=40):
    snapshots=pd.DataFrame({'machineID':[1], 'prediction_time':[T]})
    failures=pd.DataFrame({'machineID':[1]*len(event_days or []),
        'datetime':pd.to_datetime([T+pd.Timedelta(days=d) for d in (event_days or [])])})
    return outcomes(snapshots,failures,T+pd.Timedelta(days=followup))

@pytest.mark.parametrize('delay,labels,events',[
    (0,[0,0,0],[0,0,0]), (7,[1,1,1],[1]),
    (14,[0,1,1],[0,1]), (30,[0,0,1],[0,0,1]),
    (30.0001,[0,0,0],[0,0,0]), (7.0001,[0,1,1],[0,1])])
def test_exact_boundaries(delay,labels,events):
    out=landmark([delay])
    assert out[[f'label_failure_{h}d' for h in [7,14,30]]].iloc[0].tolist()==labels
    assert periods(out).event.tolist()==events

def test_first_event_only_and_simultaneous_component_failures():
    out=landmark([7,7,14,30])
    assert out.first_event_days.iloc[0]==7
    assert periods(out).interval.tolist()==[1]

@pytest.mark.parametrize('end,observed_intervals',[(7,0),(10,1),(14,1),(30,2),(31,3)])
def test_exclusive_censoring_end(end,observed_intervals):
    out=landmark([],end)
    assert len(periods(out))==observed_intervals
    assert out.complete_30d.iloc[0]==(end>30)
    if end<=30: assert np.isnan(out.label_failure_30d.iloc[0])

def test_event_before_partial_followup_counts_but_at_end_does_not():
    assert periods(landmark([8],10)).event.tolist()==[0,1]
    assert periods(landmark([10],10)).event.tolist()==[0]

def test_composition_known_answer_and_rejects_invalid():
    np.testing.assert_allclose(cumulative_probabilities([[.1,.2,.5]]),[[.1,.28,.64]])
    for value in [[[np.nan,0,0]], [[1.01,0,0]], [[0,0]]]:
        with pytest.raises(ValueError): cumulative_probabilities(value)

def test_past_only_features_include_neither_current_nor_future_events():
    machines=pd.DataFrame({'machineID':[1],'model':['model1'],'age':[4]})
    def source(days): return pd.DataFrame({'machineID':[1]*len(days), 'datetime':[T+pd.Timedelta(days=d) for d in days]})
    past=source([-8,-1]); augmented=source([-8,-1,0,1,7])
    a=machine_features(machines,past,past,past,1,T)
    b=machine_features(machines,augmented,augmented,augmented,1,T)
    pd.testing.assert_frame_equal(a,b)
    assert a.maintenance_events_prior.iloc[0]==2
    assert a.error_events_7d.iloc[0]==1

def synthetic_training():
    n=30
    f=pd.DataFrame({key:np.arange(n)%5 for key in FEATURES})
    f['machine_model']='model1'
    out=pd.DataFrame({'landmark_position':range(n),'followup_days':40.,
                      'first_event_days':[5,10,20,np.nan,np.nan]*6})
    return f,periods(out)

@pytest.mark.parametrize('algorithm',['lr','hgb'])
def test_schema_repeatability_frozen_calibration_and_loading(algorithm,tmp_path):
    f,pp=synthetic_training()
    params=dict(max_iter=15,min_samples_leaf=3,early_stopping=False)
    params.pop('early_stopping')
    model=HazardV2(algorithm,params).fit(f,pp)
    before=joblib.hash((model.preprocessor_,model.classifier_))
    model.calibrate(f,pp)  # Fixture-only; real run enforces disjoint chronological sets.
    assert before==joblib.hash((model.preprocessor_,model.classifier_))
    p=model.predict(f,calibrated=True)
    assert p.shape==(30,3) and np.isfinite(p).all()
    assert ((0<=p)&(p<=1)).all() and (np.diff(p,axis=1)>=0).all()
    np.testing.assert_array_equal(p,model.predict(f,calibrated=True))
    path=tmp_path/'test.joblib'; joblib.dump(model,path)
    np.testing.assert_array_equal(p,joblib.load(path).predict(f,calibrated=True))
    np.testing.assert_array_equal(model.predict(f),HazardV2(algorithm,params).fit(f,pp).predict(f))
    assert model.feature_schema_==FEATURES
    np.testing.assert_allclose(model.preprocessor_.named_transformers_['num'].named_steps['impute'].statistics_,
                               f[FEATURES[:-1]].median().to_numpy())
    with pytest.raises(ValueError): model.predict(f.drop(columns=FEATURES[0]))
    with pytest.raises(ValueError): model.predict(f.assign(machine_model=None))
    missing=f.copy(); missing.loc[0,FEATURES[0]]=np.nan
    assert np.isfinite(model.predict(missing)).all()
    malformed=tmp_path/'invalid.joblib'; joblib.dump({'schema':'wrong'},malformed)
    with pytest.raises(ValueError,match='schema'): load_research_model(malformed)

def test_no_post_failure_or_missing_intervals_accepted():
    f,pp=synthetic_training()
    pp=pd.concat([pp,pd.DataFrame({'landmark_position':[0],'interval':[2],'event':[0]})])
    with pytest.raises(ValueError,match='after first event'): HazardV2().fit(f,pp)

def test_purged_machine_date_groups():
    dates=pd.date_range('2015-02-01',periods=18,freq='7D')
    f=pd.DataFrame([dict(split='train',prediction_time=t,machineID=i,label_failure_7d=i%2) for t in dates for i in range(1,101)])
    for a,b,_ in development_folds(f):
        assert f.iloc[a].prediction_time.max()+pd.Timedelta(days=30)<f.iloc[b].prediction_time.min()
        assert not set(a)&set(b)
        assert f.iloc[b].groupby('prediction_time').size().eq(100).all()

def test_real_results_and_deployed_artifact_unchanged():
    import json
    folder=Path(__file__).parent/'run_001'
    summary=json.loads((folder/'summary.json').read_text())
    assert sha256(Path(__file__).parents[1]/'models/c3_7day_sigmoid.joblib')==summary['original_artifact_sha256']
    saved=pd.read_csv(folder/'heldout_predictions.csv')
    scores=pd.read_csv(folder/'comparison.csv')
    for method,g in saved.groupby('method'):
        wide=g.pivot(index=['machineID','prediction_time'],columns='horizon',values='p')
        assert len(wide)==1300 and np.isfinite(wide).all().all()
        truth=g.pivot(index=['machineID','prediction_time'],columns='horizon',values='y')
        assert (np.diff(truth.to_numpy(),axis=1)>=0).all()
        if method.startswith(('lr_','hgb_')): assert (np.diff(wide.to_numpy(),axis=1)>=0).all()
        for h,data in g.groupby('horizon'):
            ref=scores[(scores.method==method)&(scores.horizon_days==h)].iloc[0]
            assert np.mean((data.y-data.p)**2)==pytest.approx(ref.brier,abs=1e-14)
