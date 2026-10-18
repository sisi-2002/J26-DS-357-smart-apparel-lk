"""Component-owned route registration for later shared-app integration."""
from fastapi import APIRouter

from .endpoints.failure_risk import router as failure_risk_router
from .endpoints.machine_rankings import router as machine_rankings_router


router = APIRouter(prefix="/api/v1/component3", tags=["Component 3: historical research demo"])
router.include_router(failure_risk_router)
router.include_router(machine_rankings_router)
