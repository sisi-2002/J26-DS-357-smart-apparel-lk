"""Component-owned route registration for later shared-app integration."""
from fastapi import APIRouter

from .endpoints.failure_risk import router as failure_risk_router


router = APIRouter(prefix="/api/v1/component3", tags=["Component 3: historical research demo"])
router.include_router(failure_risk_router)
