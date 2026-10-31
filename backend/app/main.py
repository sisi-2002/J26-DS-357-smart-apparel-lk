"""Shared FastAPI application for inspection and predictive maintenance."""
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import Settings, settings as application_settings
from app.db import session as db_session
from app.components.component1.endpoints import router as component1_router
from app.components.component1.services import detection_service
from app.components.component3.ml.failure_model_loader import load_resources
from app.components.component3.router import router as component3_router
from app.components.component3.services.failure_prediction_service import FailurePredictionService


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    configuration = settings if settings is not None else Settings.from_env()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        logger.info("Starting up")
        application.state.component1_database_ready = False
        application.state.component3_service = None
        application.state.component3_experimental_service = None
        try:
            # Each component may be unavailable without taking the other offline.
            try:
                db_session.init_engine()
                application.state.component1_database_ready = db_session.engine is not None
                if application_settings.DATABASE_URL:
                    logger.info("Database engine initialised")
                else:
                    logger.warning("DATABASE_URL not set - database features unavailable")
            except Exception:
                logger.exception("Component 1 database initialisation failed")

            try:
                detection_service.load_model(
                    application_settings.C1_MODEL_WEIGHTS,
                    device=application_settings.C1_MODEL_DEVICE,
                    expected_sha256=application_settings.C1_MODEL_EXPECTED_SHA256,
                    freeze_manifest_path=application_settings.C1_MODEL_FREEZE_MANIFEST,
                )
                if not detection_service.is_loaded:
                    logger.warning("Component 1 model unavailable: %s", detection_service.load_error)
            except Exception:
                logger.exception("Component 1 model initialisation failed")

            try:
                resources = load_resources(configuration)
                application.state.component3_service = FailurePredictionService(resources)
            except Exception:
                logger.exception("Component 3 unavailable: check C3_MODEL_PATH and C3_DATA_DIR")

            if configuration.c3_experimental_multihorizon:
                try:
                    from app.components.component3.experimental.service import ExperimentalPredictionService
                    application.state.component3_experimental_service = ExperimentalPredictionService.load(configuration)
                except Exception:
                    logger.exception("Experimental multi-horizon unavailable; seven-day prediction remains independent")
            yield
        finally:
            application.state.component3_service = None
            application.state.component3_experimental_service = None
            application.state.component1_database_ready = False
            logger.info("Shutting down")
            await db_session.dispose_engine()

    application = FastAPI(
        title="SMART APPAREL-LK",
        description="Apparel defect inspection and explainability, plus predictive maintenance. "
        "Component 3 uses the historical synthetic Microsoft Azure benchmark and is not "
        "validated for real apparel-factory machinery.",
        version="0.1.0",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173", "http://localhost:3000",
            "http://127.0.0.1:5173", "http://127.0.0.1:3000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # Preserve the Component 1 local-demo auth placeholder.
    # Mount the platform JWT middleware here when it becomes available.
    if application_settings.C1_AUTH_DISABLED:
        logger.warning("AUTH DISABLED - local demo mode only. Do not deploy with C1_AUTH_DISABLED=true.")

    application.include_router(component1_router)
    application.include_router(component3_router)
    if configuration.c3_experimental_multihorizon:
        from app.components.component3.experimental.router import router as experimental_router
        application.include_router(experimental_router)

    @application.get("/api/v1/health", tags=["Platform"])
    def health(request: Request):
        # Preserve the existing Component 3 probe's status-code contract.
        # Component 1 also retains its own health and readiness routes.
        ready = request.app.state.component3_service is not None
        return JSONResponse(
            {"status": "ready" if ready else "unavailable", "component3_ready": ready,
             "component1_ready": detection_service.is_loaded,
             "component1_database_ready": request.app.state.component1_database_ready,
             "data_status": "synthetic_azure_historical_demo"},
            status_code=200 if ready else 503,
        )

    return application


app = create_app()
