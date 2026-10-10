"""FastAPI application for SMART APPAREL-LK."""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.db.session import init_engine, dispose_engine
from app.components.component1.endpoints import router as c1_router
from app.components.component1.services import detection_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up")
    init_engine()
    if settings.DATABASE_URL:
        logger.info("Database engine initialised")
    else:
        logger.warning("DATABASE_URL not set — database features unavailable")
    detection_service.load_model(
        settings.C1_MODEL_WEIGHTS,
        device=settings.C1_MODEL_DEVICE,
        expected_sha256=settings.C1_MODEL_EXPECTED_SHA256,
        freeze_manifest_path=settings.C1_MODEL_FREEZE_MANIFEST,
    )
    if not detection_service.is_loaded:
        logger.warning(
            "Model not loaded — health endpoint will return 503. Reason: %s",
            detection_service.load_error,
        )
    yield
    logger.info("Shutting down")
    await dispose_engine()


app = FastAPI(
    title="SMART APPAREL-LK",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — local development only
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Auth placeholder ---
# When the platform team provides shared JWT middleware, mount it here:
#   from app.core.auth import JWTMiddleware
#   app.add_middleware(JWTMiddleware)
# Until then, C1_AUTH_DISABLED=true skips auth (local demo only).
if settings.C1_AUTH_DISABLED:
    logger.warning("AUTH DISABLED — local demo mode only. Do not deploy with C1_AUTH_DISABLED=true.")

app.include_router(c1_router)
