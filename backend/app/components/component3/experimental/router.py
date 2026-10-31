"""Registered only when the backend experimental flag is explicitly enabled."""
import logging
from typing import Literal
from uuid import uuid4
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, NaiveDatetime, model_validator
from ..schemas.failure_schema import FailureRiskRequest, HistoricalPredictionRequest, EvidenceProfile
from .service import ExperimentalPredictionFailed

logger = logging.getLogger(__name__)
router = APIRouter(prefix='/api/v1/component3/experimental', tags=['Experimental multi-horizon preview'])


class Probabilities(BaseModel):
    p7: float = Field(ge=0, le=1, allow_inf_nan=False)
    p14: float = Field(ge=0, le=1, allow_inf_nan=False)
    p30: float = Field(ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode='after')
    def ordered(self):
        if not self.p7 <= self.p14 <= self.p30:
            raise ValueError('Experimental probabilities must be ordered')
        return self


class ExperimentalMetadata(BaseModel):
    model_version: str
    model_release: Literal['v3-run_003-unpooled-hgb-sigmoid']
    model_schema: Literal['unpooled-inclusive-v3']
    experimental: Literal[True]
    research_only: Literal[True]
    data_status: Literal['synthetic_azure_historical_demo']
    interval_semantics: str
    note: str
    request_id: str


class DescriptiveEvidence(EvidenceProfile):
    error_events_prior: int = Field(ge=0)
    error_events_7d: int = Field(ge=0)
    error_events_30d: int = Field(ge=0)
    error_events_90d: int = Field(ge=0)
    days_since_last_error: float | None = Field(ge=0, allow_inf_nan=False)


class ExperimentalRiskResponse(Probabilities, ExperimentalMetadata):
    machine_id: int
    prediction_time: NaiveDatetime
    evidence_profile: DescriptiveEvidence


class RankingRequest(HistoricalPredictionRequest):
    horizon_days: Literal[7, 14, 30] = 7


class RankedPrediction(Probabilities):
    rank: int = Field(ge=1, le=10)
    machine_id: int = Field(ge=1, le=100)


class ExperimentalRankingResponse(ExperimentalMetadata):
    prediction_time: NaiveDatetime
    horizon_days: Literal[7, 14, 30]
    scored_machine_count: Literal[100]
    returned_machine_count: Literal[10]
    machines: list[RankedPrediction] = Field(min_length=10, max_length=10)


@router.post('/predict-risk', response_model=ExperimentalRiskResponse)
def predict_experimental(body: FailureRiskRequest, request: Request):
    service = getattr(request.app.state, 'component3_experimental_service', None)
    if service is None:
        raise HTTPException(503, 'Experimental multi-horizon model unavailable; check experimental configuration. No substitute prediction is returned.')
    try:
        result = service.predict(body.machine_id, body.prediction_time.isoformat())
    except KeyError as exc:
        raise HTTPException(404, 'Machine not found in the synthetic benchmark') from exc
    except ExperimentalPredictionFailed as exc:
        logger.exception('Experimental multi-horizon inference failed')
        raise HTTPException(500, 'Experimental multi-horizon prediction failed; no substitute prediction is returned') from exc
    return ExperimentalRiskResponse(**result, request_id=str(uuid4()))


@router.post('/top-risk-machines', response_model=ExperimentalRankingResponse)
def rank_experimental(body: RankingRequest, request: Request):
    service = getattr(request.app.state, 'component3_experimental_service', None)
    if service is None:
        raise HTTPException(503, 'Experimental ranking unavailable; check experimental configuration. No substitute ranking is returned.')
    try:
        result = service.rank(body.prediction_time.isoformat(), body.horizon_days)
    except ExperimentalPredictionFailed as exc:
        logger.exception('Experimental multi-horizon ranking failed')
        raise HTTPException(500, 'Experimental ranking failed; no substitute ranking is returned') from exc
    return ExperimentalRankingResponse(**result, request_id=str(uuid4()))
