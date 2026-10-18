"""Historical benchmark prediction endpoint."""
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request

from ..schemas.failure_schema import FailureRiskRequest, FailureRiskResponse
from ..services.failure_prediction_service import (
    FailurePredictionService, MachineNotFound, PredictionFailed,
)


router = APIRouter()


def get_prediction_service(request: Request) -> FailurePredictionService:
    service = getattr(request.app.state, "component3_service", None)
    if service is None:
        raise HTTPException(503, "Component 3 inference unavailable; check configured model and benchmark data")
    return service


@router.post(
    "/predict-risk", response_model=FailureRiskResponse,
    summary="Predict historical 7-day risk on the synthetic Azure benchmark",
    description="Research demonstration only. Returns model risk and evidence characteristics; "
    "does not provide apparel-factory predictions, reliability scores or maintenance actions.",
    responses={404: {"description": "Unknown benchmark machine"},
               503: {"description": "Model or benchmark sources unavailable"},
               500: {"description": "Inference failed"}},
)
def predict_risk(
    body: FailureRiskRequest,
    service: Annotated[FailurePredictionService, Depends(get_prediction_service)],
) -> FailureRiskResponse:
    try:
        result = service.predict(body.machine_id, body.prediction_time.isoformat())
    except MachineNotFound as exc:
        raise HTTPException(404, str(exc)) from exc
    except PredictionFailed as exc:
        raise HTTPException(500, "Component 3 prediction failed") from exc
    return FailureRiskResponse(**result, request_id=str(uuid4()))
