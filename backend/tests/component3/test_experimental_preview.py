from dataclasses import replace
from copy import copy
import sys
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import HistGradientBoostingClassifier
from app.core.config import REPOSITORY_ROOT, Settings
from app.main import create_app
from app.components.component3.experimental.service import ExperimentalPredictionService, _load_bundle

URL='/api/v1/component3/experimental/predict-risk'
BODY={'machine_id':1,'prediction_time':'2015-09-02T06:00:00'}
ARTIFACT=REPOSITORY_ROOT/'research/component3/experimental_multihorizon_v3/run_003/unpooled_hgb_research.joblib'

@pytest.fixture(autouse=True)
def no_booster_training(monkeypatch):
    def forbidden(*args,**kwargs): raise AssertionError('No HGB training during serving')
    monkeypatch.setattr(HistGradientBoostingClassifier,'fit',forbidden)

@pytest.fixture
def enabled_settings(local_settings):
    return replace(local_settings,c3_experimental_multihorizon=True,c3_experimental_model_path=ARTIFACT)

@pytest.fixture
def experimental_client(enabled_settings):
    with TestClient(create_app(enabled_settings)) as client: yield client

def test_off_by_default_no_route_or_loading(fixture_settings,monkeypatch):
    assert Settings().c3_experimental_multihorizon is False
    def forbidden(*args): raise AssertionError('Disabled experimental loader called')
    monkeypatch.setattr(ExperimentalPredictionService,'load',forbidden)
    with TestClient(create_app(fixture_settings)) as client:
        assert client.post(URL,json=BODY).status_code==404
        assert URL not in client.get('/openapi.json').json()['paths']
        assert client.post('/api/v1/component3/predict-risk',json=BODY).json()['predicted_failure_probability']==.25

def test_flags_parse_explicitly(monkeypatch):
    monkeypatch.delenv('C3_EXPERIMENTAL_MULTIHORIZON',raising=False)
    assert not Settings.from_env().c3_experimental_multihorizon
    monkeypatch.setenv('C3_EXPERIMENTAL_MULTIHORIZON','true')
    assert Settings.from_env().c3_experimental_multihorizon
    monkeypatch.setenv('C3_EXPERIMENTAL_MULTIHORIZON','false')
    assert not Settings.from_env().c3_experimental_multihorizon

def test_real_api_across_dates_matches_saved_predictions_exactly(experimental_client):
    saved=pd.read_csv(ARTIFACT.parent/'heldout_predictions.csv',float_precision='round_trip')
    saved=saved[saved.method.eq('unpooled_hgb')]
    for timestamp in sorted(saved.prediction_time.unique()):
        for mid in range(1,101):
            body={'machine_id':mid,'prediction_time':timestamp.replace(' ','T')}
            response=experimental_client.post(URL,json=body)
            assert response.status_code==200,response.text
            result=response.json()
            ref=saved[(saved.machineID==mid)&(saved.prediction_time==timestamp)].sort_values('horizon').p.to_numpy()
            np.testing.assert_array_equal([result['p7'],result['p14'],result['p30']],ref)
            assert result['experimental'] and result['research_only']
            assert result['p7']<=result['p14']<=result['p30']
            assert result['model_version'].startswith('sha256:')
            assert result['model_release']=='v3-run_003-unpooled-hgb-sigmoid'
            assert '(0,7]' in result['interval_semantics']
            service=experimental_client.app.state.component3_experimental_service
            row=service.inference.machine_features(*service.sources,mid,body['prediction_time']).iloc[0]
            assert result['evidence_profile']==service._evidence(row)
    a=experimental_client.post(URL,json=BODY).json();b=experimental_client.post(URL,json=BODY).json()
    assert a.pop('request_id')!=b.pop('request_id');assert a==b

def test_full_1300_feature_schema_and_artifact_roundtrip(experimental_client):
    service=experimental_client.app.state.component3_experimental_service
    saved=pd.read_csv(ARTIFACT.parent/'heldout_predictions.csv',parse_dates=['prediction_time'],float_precision='round_trip')
    wide=saved[saved.method.eq('unpooled_hgb')].pivot(index=['machineID','prediction_time'],columns='horizon',values='p')
    frame=pd.concat([service.inference.machine_features(*service.sources,int(mid),t) for mid,t in wide.index],ignore_index=True)
    assert list(service.model.models_[0].feature_names_in_)==service.inference.FEATURE_COLUMNS
    np.testing.assert_array_equal(service.model.predict(frame),wide[[7,14,30]].to_numpy())
    assert np.isfinite(service.model.predict(frame)).all()

@pytest.mark.parametrize('change',[{'machine_id':'1'},{'machine_id':True},{'machine_id':0},
    {'prediction_time':'2026-10-22T06:00:00'},{'prediction_time':'2015-09-03T06:00:00'},
    {'prediction_time':'2015-09-02T06:00:00Z'},{'model_path':'other.joblib'},{'line':1}])
def test_input_validation(experimental_client,change):
    assert experimental_client.post(URL,json={**BODY,**change}).status_code==422

def test_unknown_machine(experimental_client):
    assert experimental_client.post(URL,json={**BODY,'machine_id':101}).status_code==404

@pytest.mark.parametrize('missing',['model','data','unconfigured'])
def test_unavailable_is_independent_of_original_model(enabled_settings,tmp_path,missing):
    settings=enabled_settings
    if missing=='model':settings=replace(settings,c3_experimental_model_path=tmp_path/'absent')
    elif missing=='data':settings=replace(settings,c3_data_dir=tmp_path/'absent')
    else:settings=replace(settings,c3_experimental_model_path=None)
    with TestClient(create_app(settings)) as client:
        response=client.post(URL,json=BODY)
        assert response.status_code==503 and 'p7' not in response.json()
        assert str(tmp_path) not in response.text
        if missing!='data': assert client.post('/api/v1/component3/predict-risk',json=BODY).status_code==200

def test_wrong_artifact_fails_before_deserialization(enabled_settings,tmp_path,monkeypatch):
    path=tmp_path/'wrong.joblib';path.write_bytes(b'wrong artifact')
    def forbidden(*args):raise AssertionError('Untrusted payload deserialized')
    monkeypatch.setattr('app.components.component3.experimental.service._load_bundle',forbidden)
    with pytest.raises(ValueError,match='identity'):
        ExperimentalPredictionService.load(replace(enabled_settings,c3_experimental_model_path=path))


@pytest.mark.parametrize('defect',['schema','features','algorithm','calibrators','preprocessing'])
def test_incompatible_bundle_fails_safely(enabled_settings,monkeypatch,defect):
    bundle, model_class = _load_bundle(ARTIFACT.read_bytes())
    model = copy(bundle['model'])
    bundle = {**bundle, 'model': model}
    if defect == 'schema': bundle['schema'] = 'wrong'
    elif defect == 'features': bundle['features'] = ['wrong']
    elif defect == 'algorithm': model.algorithm = 'lr'
    elif defect == 'calibrators': model.calibrators_ = []
    else: model.feature_schema_ = ['wrong']
    monkeypatch.setattr('app.components.component3.experimental.service._load_bundle',lambda payload:(bundle,model_class))
    with TestClient(create_app(enabled_settings)) as client:
        assert client.post(URL,json=BODY).status_code == 503
        assert client.post('/api/v1/component3/predict-risk',json=BODY).status_code == 200


def test_changed_data_fails_before_loading(enabled_settings,tmp_path,monkeypatch):
    (tmp_path/'PdM_errors.csv').write_text('changed benchmark')
    def forbidden(*args): raise AssertionError('Model loaded despite changed data')
    monkeypatch.setattr('app.components.component3.experimental.service._load_bundle',forbidden)
    with pytest.raises(ValueError,match='benchmark data mismatch'):
        ExperimentalPredictionService.load(replace(enabled_settings,c3_data_dir=tmp_path))

@pytest.mark.parametrize('value',[[.6,.5,.7],[np.nan,.5,.7],[-.1,.5,.7],[.1,.5,1.1]])
def test_bad_outputs_fail_without_fallback(experimental_client,monkeypatch,value):
    service=experimental_client.app.state.component3_experimental_service
    monkeypatch.setattr(service.model,'predict',lambda *a,**k:np.array([value]))
    result=experimental_client.post(URL,json=BODY)
    assert result.status_code==500 and 'p7' not in result.json()
    assert experimental_client.post('/api/v1/component3/predict-risk',json=BODY).status_code==200

def test_loader_restores_registry_and_search_path(local_settings):
    names=['build_and_evaluate','hazard_model','hazard_v2','methods_v3']
    before={n:sys.modules.get(n) for n in names};paths=sys.path[:];bytecode=sys.dont_write_bytecode
    _load_bundle(ARTIFACT.read_bytes())
    assert sys.path==paths and sys.dont_write_bytecode==bytecode
    assert {n:sys.modules.get(n) for n in names}==before

def test_enabled_disabled_existing_api_outputs_unchanged(local_settings,enabled_settings):
    with TestClient(create_app(replace(local_settings,c3_experimental_multihorizon=False))) as off:
        prediction=off.post('/api/v1/component3/predict-risk',json=BODY).json()
        ranking=off.post('/api/v1/component3/top-risk-machines',json={'prediction_time':BODY['prediction_time']}).json()
    with TestClient(create_app(enabled_settings)) as on:
        for route,body,expected in [('/predict-risk',BODY,prediction),('/top-risk-machines',{'prediction_time':BODY['prediction_time']},ranking)]:
            actual=on.post('/api/v1/component3'+route,json=body).json()
            actual.pop('request_id');expected.pop('request_id');assert actual==expected


RANK_URL='/api/v1/component3/experimental/top-risk-machines'

def test_all_dates_and_horizons_actual_v3_top10(experimental_client):
    saved=pd.read_csv(ARTIFACT.parent/'heldout_predictions.csv',float_precision='round_trip')
    saved=saved[saved.method.eq('unpooled_hgb')]
    for timestamp in sorted(saved.prediction_time.unique()):
        for horizon in [7,14,30]:
            body={'prediction_time':timestamp.replace(' ','T'),'horizon_days':horizon}
            result=experimental_client.post(RANK_URL,json=body)
            assert result.status_code==200,result.text
            data=result.json()
            expected=saved[(saved.prediction_time==timestamp)&(saved.horizon==horizon)].sort_values(['p','machineID'],ascending=[False,True]).head(10)
            assert [r['machine_id'] for r in data['machines']]==expected.machineID.tolist()
            np.testing.assert_array_equal([r[f'p{horizon}'] for r in data['machines']],expected.p.to_numpy())
            assert data['scored_machine_count']==100 and data['horizon_days']==horizon
            top=data['machines'][0]
            prediction=experimental_client.post(URL,json={'machine_id':top['machine_id'],'prediction_time':body['prediction_time']}).json()
            assert [top[k] for k in ['p7','p14','p30']]==[prediction[k] for k in ['p7','p14','p30']]
            assert data['model_version']==prediction['model_version']
    assert len(experimental_client.app.state.component3_experimental_service._batch_cache)==13


def test_batch_is_one_call_per_conditional_model_and_cache_is_shared(experimental_client,monkeypatch):
    from unittest.mock import Mock
    service=experimental_client.app.state.component3_experimental_service
    spies=[]
    for model in service.model.calibrators_:
        spy=Mock(wraps=model.predict_proba);monkeypatch.setattr(model,'predict_proba',spy);spies.append(spy)
    for horizon in [7,14,30,7]:
        response=experimental_client.post(RANK_URL,json={'prediction_time':BODY['prediction_time'],'horizon_days':horizon})
        assert response.status_code==200
    for spy in spies:
        assert spy.call_count==1 and len(spy.call_args.args[0])==100
    # Responses cannot mutate the immutable cached triplets.
    response.json()['machines'][0]['p7']=1
    assert experimental_client.post(RANK_URL,json={'prediction_time':BODY['prediction_time']}).json()['machines'][0]['p7']!=1


def test_batch_ties_use_machine_id(experimental_client,monkeypatch):
    service=experimental_client.app.state.component3_experimental_service
    monkeypatch.setattr(service.model,'predict',lambda *a,**k: np.tile([.1,.2,.3],(100,1)))
    for h in [7,14,30]:
        data=experimental_client.post(RANK_URL,json={'prediction_time':BODY['prediction_time'],'horizon_days':h}).json()
        assert [r['machine_id'] for r in data['machines']]==list(range(1,11))


@pytest.mark.parametrize('change',[{'horizon_days':1},{'horizon_days':'7'},{'prediction_time':'2026-01-01T06:00:00'},{'machine_id':1}])
def test_batch_request_validation(experimental_client,change):
    assert experimental_client.post(RANK_URL,json={'prediction_time':BODY['prediction_time'],**change}).status_code==422


def test_batch_failure_does_not_cache_invalid_values(experimental_client,monkeypatch):
    service=experimental_client.app.state.component3_experimental_service
    monkeypatch.setattr(service.model,'predict',lambda *a,**k: np.tile([.5,.2,.3],(100,1)))
    assert experimental_client.post(RANK_URL,json={'prediction_time':BODY['prediction_time']}).status_code==500
    assert service._batch_cache=={}
    assert experimental_client.post('/api/v1/component3/top-risk-machines',json={'prediction_time':BODY['prediction_time']}).status_code==200


def test_batch_off_and_missing_model(fixture_settings,enabled_settings,tmp_path):
    with TestClient(create_app(fixture_settings)) as c:
        assert c.post(RANK_URL,json={'prediction_time':BODY['prediction_time']}).status_code==404
    with TestClient(create_app(replace(enabled_settings,c3_experimental_model_path=tmp_path/'absent'))) as c:
        assert c.post(RANK_URL,json={'prediction_time':BODY['prediction_time']}).status_code==503


def test_null_error_history_and_strict_past_evidence(experimental_client):
    service=experimental_client.app.state.component3_experimental_service
    features=service.inference.machine_features(*service.sources,1,BODY['prediction_time'])
    past=(service.sources[0],*(frame[frame.datetime<pd.Timestamp(BODY['prediction_time'])].copy() for frame in service.sources[1:]))
    actual=service.inference.machine_features(*past,1,BODY['prediction_time'])
    pd.testing.assert_frame_equal(features,actual)
    row=features.iloc[0].copy();row['days_since_last_error']=np.nan;row['error_events_prior']=0
    assert service._evidence(row)['days_since_last_error'] is None
