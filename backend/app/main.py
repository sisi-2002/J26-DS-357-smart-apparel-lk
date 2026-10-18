"""Minimal shared scaffold with isolated Component 3 initialization."""
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.config import Settings
from app.components.component3.ml.failure_model_loader import load_resources
from app.components.component3.router import router as component3_router
from app.components.component3.services.failure_prediction_service import FailurePredictionService


logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.component3_service = None
        try:
            resources = load_resources(settings if settings is not None else Settings.from_env())
            application.state.component3_service = FailurePredictionService(resources)
        except Exception:
            # Keep docs and future components available. No automatic model export.
            logger.exception("Component 3 unavailable: check C3_MODEL_PATH and C3_DATA_DIR")
        yield
        application.state.component3_service = None

    application = FastAPI(
        title="SMART APPAREL LK — Component 3 research demo",
        description="Historical synthetic Microsoft Azure benchmark inference. "
        "Not validated for real apparel-factory machinery.",
        version="0.1.0", lifespan=lifespan,
    )
    application.include_router(component3_router)

    @application.get("/api/v1/health", tags=["Platform"])
    def health(request: Request):
        ready = request.app.state.component3_service is not None
        return JSONResponse(
            {"status": "ready" if ready else "unavailable", "component3_ready": ready,
             "data_status": "synthetic_azure_historical_demo"},
            status_code=200 if ready else 503,
        )

    return application


app = create_app()
