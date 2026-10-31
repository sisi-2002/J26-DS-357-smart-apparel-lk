import os,sys
os.environ['OMP_NUM_THREADS']='1'
sys.dont_write_bytecode=True
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize
from methods_v3 import pav,anchored_pav,UnpooledHazard,FEATURES,periods,outcomes


def test_pav_known_block_pooling():
    p=np.array([[.9,.8,.1],[.1,.9,.2],[.8,.1,.9],[.1,.2,.3]])
    np.testing.assert_allclose(pav(p),[[.6,.6,.6],[.1,.55,.55],[.45,.45,.9],[.1,.2,.3]])
    np.testing.assert_allclose(anchored_pav(p),[[.9,.9,.9],[.1,.55,.55],[.8,.8,.9],[.1,.2,.3]])
    np.testing.assert_array_equal(anchored_pav(p)[:,0],p[:,0])

@pytest.mark.parametrize('anchored',[False,True])
def test_projection_optimality_against_independent_quadratic_optimizer(anchored):
    for p in np.random.default_rng(42).random((40,3)):
        result=(anchored_pav if anchored else pav)(p[None,:])[0]
        constraints=[{'type':'ineq','fun':lambda x:x[1]-x[0]},
                     {'type':'ineq','fun':lambda x:x[2]-x[1]}]
        if anchored: constraints.append({'type':'eq','fun':lambda x:x[0]-p[0]})
        optimum=minimize(lambda x:np.sum((x-p)**2),np.repeat(p[0] if anchored else .5,3),
            method='SLSQP',bounds=[(0,1)]*3,constraints=constraints,options={'ftol':1e-13,'maxiter':200})
        assert optimum.success
        assert np.sum((result-p)**2)==pytest.approx(optimum.fun,abs=1e-10)
        assert (np.diff(result)>=0).all()

def test_standard_total_loss_guarantee_for_every_complete_nested_binary_target():
    p=np.random.default_rng(1).random((10000,3)); projected=pav(p)
    for target in [[0,0,0],[0,0,1],[0,1,1],[1,1,1]]:
        assert (((projected-target)**2).sum(axis=1)<=((p-target)**2).sum(axis=1)+1e-13).all()

def test_guarantee_is_not_per_horizon_and_does_not_extend_to_anchor():
    p=np.array([[.9,.8,.1]])
    assert ((pav(p)[0,0]-1)**2) > ((p[0,0]-1)**2)
    assert (anchored_pav(p)**2).sum() > (p**2).sum()  # Complete nested outcome 000.

@pytest.mark.parametrize('p',[[[np.nan,0,1]],[[0,-.1,1]],[[0,1]],[[0,1,1.1]]])
def test_bad_scores_fail_explicitly(p):
    with pytest.raises(ValueError):pav(p)
    with pytest.raises(ValueError):anchored_pav(p)

def test_projection_does_not_mutate_inputs_and_handles_bounds_and_ties():
    p=np.array([[1.,0,0],[0,1,0],[0,0,0],[1,1,1]])
    copy=p.copy();a=pav(p);b=anchored_pav(p)
    np.testing.assert_array_equal(p,copy)
    for result in [a,b]:
        assert ((result>=0)&(result<=1)).all()
        np.testing.assert_array_equal(result,pav(result))

def fixture():
    n=60
    frame=pd.DataFrame({k:np.arange(n)%7 for k in FEATURES})
    frame['machine_model']='model1'
    out=pd.DataFrame({'landmark_position':range(n),'followup_days':40.,
                      'first_event_days':[7,14,30,np.nan,np.nan]*12})
    return frame,periods(out)

@pytest.mark.parametrize('algorithm',['lr','hgb'])
def test_models_use_risk_sets_frozen_calibration_repeatability_and_loading(algorithm,tmp_path):
    f,risk=fixture();params={'max_iter':10,'min_samples_leaf':4}
    model=UnpooledHazard(algorithm,params).fit(f,risk)
    before=joblib.hash(model.models_);model.calibrate(f,risk)
    assert before==joblib.hash(model.models_)
    p=model.predict(f)
    np.testing.assert_array_equal(p[:,0],model.calibrators_[0].predict_proba(f[FEATURES])[:,1])
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all() and (np.diff(p,axis=1)>=0).all()
    np.testing.assert_array_equal(p,model.predict(f))
    path=tmp_path/'fixture.joblib';joblib.dump(model,path)
    np.testing.assert_array_equal(p,joblib.load(path).predict(f))
    for index,base in enumerate(model.models_,1):
        positions=risk[risk.interval.eq(index)].landmark_position.to_numpy(dtype=int)
        np.testing.assert_allclose(base.named_steps['transform'].named_transformers_['num'].named_steps['impute'].statistics_,
                                   f.iloc[positions][FEATURES[:-1]].median())
    with pytest.raises(ValueError):model.predict(f.drop(columns=FEATURES[0]))

def test_zero_later_hazards_do_not_introduce_roundoff_order_violations():
    model=UnpooledHazard()
    model.predict_hazards=lambda f,c:np.array([[.1,0,0],[.7,0,1]])
    np.testing.assert_array_equal(model.predict(None),[[.1,.1,.1],[.7,.7,1]])

def test_saved_results_cohort_metric_and_reference_preservation():
    folder=Path(__file__).parent/'run_002'
    data=pd.read_csv(folder/'heldout_predictions.csv',float_precision='round_trip')
    scores=pd.read_csv(folder/'comparison.csv')
    matrices={}
    for name,g in data.groupby('method'):
        p=g.pivot(index=['machineID','prediction_time'],columns='horizon',values='p')
        y=g.pivot(index=['machineID','prediction_time'],columns='horizon',values='y')
        assert len(p)==1300 and p.index.equals(y.index)
        assert (np.diff(y,axis=1)>=0).all()
        if name!='independent_lr':assert (np.diff(p,axis=1)>=0).all()
        for h in [7,14,30]:
            row=scores[(scores.method==name)&(scores.horizon_days==h)].iloc[0]
            assert np.mean((y[h]-p[h])**2)==pytest.approx(row.brier,abs=1e-14)
        matrices[name]=(p,y)
    ref,y=matrices['independent_lr'];std,_=matrices['standard_pav'];anchor,_=matrices['anchored_pav'];unpooled,_=matrices['unpooled_lr']
    np.testing.assert_array_equal(anchor[7],ref[7]);np.testing.assert_array_equal(unpooled[7],ref[7])
    assert (((std-y)**2).sum(axis=1)<=((ref-y)**2).sum(axis=1)+1e-13).all()
