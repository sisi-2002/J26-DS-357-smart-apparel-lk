"""One HTTP request and one saved-model call for a complete benchmark ranking."""
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException

from .failure_risk import get_prediction_service
from ..ml.failure_ranking import IncompleteBenchmarkError
from ..schemas.failure_schema import HistoricalPredictionRequest, TopRiskMachinesResponse
from ..services.failure_prediction_service import FailurePredictionService, PredictionFailed


router = APIRouter()


@router.post(
    "/top-risk-machines", response_model=TopRiskMachinesResponse,
    summary="Top 10 relative 7-day failure probabilities among 100 synthetic machines",
    description="Scores all 100 benchmark machines in one batch; ranks full-precision probabilities. "
                "Historical synthetic research demonstration only; no maintenance recommendations.",
    responses={503: {"description": "Model/data unavailable or incomplete benchmark"},
               500: {"description": "Batch inference failed"}},
)
def top_risk_machines(
    body: HistoricalPredictionRequest,
    service: Annotated[FailurePredictionService, Depends(get_prediction_service)],
) -> TopRiskMachinesResponse:
    try:
        result = service.rank_machines(body.prediction_time.isoformat())
    except IncompleteBenchmarkError as exc:
        raise HTTPException(503, str(exc)) from exc
    except PredictionFailed as exc:
        raise HTTPException(500, "Component 3 ranking failed") from exc
    return TopRiskMachinesResponse(**result, request_id=str(uuid4()))
