"""Research-only horizon projection and interval-specific conditional models."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experimental_multihorizon_v2'))
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.utils.validation import check_is_fitted
from hazard_v2 import FEATURES, HORIZONS, outcomes, periods, require, make_baseline, cumulative_probabilities


def validate_triplets(values):
    p=np.asarray(values,dtype=float)
    require(p.ndim==2 and p.shape[1]==3 and np.isfinite(p).all(), 'Expected finite probability triplets')
    require(((p>=0)&(p<=1)).all(),'Probabilities outside [0,1]')
    return p


def pav(values):
    """Equal-weight Euclidean projection onto 0<=x1<=x2<=x3<=1; no labels."""
    p=validate_triplets(values)
    projected=np.empty_like(p)
    for i,row in enumerate(p):
        blocks=[]
        for value in row:
            blocks.append([float(value),1])
            while len(blocks)>1 and blocks[-2][0]/blocks[-2][1]>blocks[-1][0]/blocks[-1][1]:
                right=blocks.pop(); left=blocks.pop()
                blocks.append([left[0]+right[0],left[1]+right[1]])
        projected[i]=[total/count for total,count in blocks for _ in range(count)]
    return projected


def anchored_pav(values):
    """Fix x1=p1; project (p2,p3) onto p1<=x2<=x3<=1."""
    p=validate_triplets(values)
    result=p.copy()
    # Pool the two unconstrained tail coordinates if reversed, THEN apply floor.
    # Clipping each tail input first would solve a different optimization problem.
    reversed_tail=p[:,1]>p[:,2]
    mean=(p[:,1]+p[:,2])/2
    result[reversed_tail,1:]=mean[reversed_tail,None]
    result[:,1:]=np.maximum(result[:,1:],p[:,0,None])
    return result


class UnpooledHazard(BaseEstimator):
    """Three separate conditional classifiers; calibration uses later risk sets."""
    def __init__(self,algorithm='lr',parameters=None):
        self.algorithm=algorithm; self.parameters=parameters

    @staticmethod
    def features(frame):
        require(set(FEATURES).issubset(frame),'Missing historical feature columns')
        require(not frame[['machine_model']].isna().any().any(),'Missing machine category')
        require(not np.isinf(frame[FEATURES[:-1]].to_numpy(dtype=float)).any(),'Infinite features')
        return frame[FEATURES]

    def fit(self,frame,pp):
        self.features(frame)
        # Reuse the tested no-post-failure / contiguous-risk-set guard.
        from hazard_model import PooledHazardModel
        PooledHazardModel._validate_periods(pp,len(frame))
        self.models_=[]
        for interval in [1,2,3]:
            risk=pp[pp.interval.eq(interval)]
            require(risk.event.nunique()==2,'Conditional training set needs both outcomes')
            estimator=make_baseline()
            if self.algorithm=='hgb':
                processor=clone(estimator.named_steps['transform']); processor.set_params(sparse_threshold=0.)
                estimator=Pipeline([('transform',processor),('classifier',HistGradientBoostingClassifier(
                    early_stopping=False,random_state=42,**(self.parameters or {})))])
            else: require(self.algorithm=='lr','Unknown classifier')
            estimator.fit(self.features(frame).iloc[risk.landmark_position.to_numpy(dtype=int)],risk.event.to_numpy())
            self.models_.append(estimator)
        self.feature_schema_=list(FEATURES)
        if hasattr(self,'calibrators_'): del self.calibrators_
        return self

    def calibrate(self,frame,pp):
        check_is_fitted(self,'models_')
        from hazard_model import PooledHazardModel
        PooledHazardModel._validate_periods(pp,len(frame))
        self.calibrators_=[]
        for interval,estimator in enumerate(self.models_,1):
            risk=pp[pp.interval.eq(interval)]
            require(risk.event.nunique()==2,'Conditional calibration set needs both outcomes')
            frozen=CalibratedClassifierCV(FrozenEstimator(estimator),method='sigmoid')
            frozen.fit(self.features(frame).iloc[risk.landmark_position.to_numpy(dtype=int)],risk.event.to_numpy())
            self.calibrators_.append(frozen)
        return self

    def predict_hazards(self,frame,calibrated=True):
        attr='calibrators_' if calibrated else 'models_'
        check_is_fitted(self,attr)
        return np.column_stack([model.predict_proba(self.features(frame))[:,1] for model in getattr(self,attr)])

    def predict(self,frame,calibrated=True):
        h=self.predict_hazards(frame,calibrated)
        validate_triplets(h)
        p=np.empty_like(h)
        p[:,0]=h[:,0]
        # Equivalent survival accumulation; avoids complement cancellation when
        # later hazards are zero, preserving P7 and ordering in floating point.
        p[:,1]=p[:,0]+(1-p[:,0])*h[:,1]
        p[:,2]=p[:,1]+(1-p[:,1])*h[:,2]
        return p
