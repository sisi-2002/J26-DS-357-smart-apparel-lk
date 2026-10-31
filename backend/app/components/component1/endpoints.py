"""Component 1 API endpoints."""
import logging
import uuid
from datetime import datetime, timezone

import cv2
import numpy as np
from fastapi import APIRouter, Depends, Query, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.components.component1 import repositories as repo
from app.components.component1.model_identity import resolve_inspection_experiment_id
from app.components.component1.models import DefectDetection, DefectExplanation, DefectInspection
from app.components.component1.schemas import (
    BboxResponse,
    DetectionResponse,
    ErrorResponse,
    ExplanationRequest,
    ExplanationResponse,
    HealthResponse,
    InspectionListItem,
    InspectionListResponse,
    InspectionResponse,
    InspectionSummary,
    ModelIdentity,
    ReadinessResponse,
)
from app.components.component1.services import detection_service
from app.components.component1.storage import (
    ImageStorage,
    ImageValidationError,
    compute_hash,
    validate_image_bytes,
)
from app.core.config import settings
from app.db.session import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/component1", tags=["component1"])

_storage: ImageStorage | None = None


def _get_storage() -> ImageStorage:
    global _storage
    if _storage is None:
        _storage = ImageStorage(settings.C1_STORAGE_DIR)
    return _storage


def _artifact_url(relative_path: str) -> str:
    return f"/api/v1/component1/artifacts/{relative_path}"


def _build_inspection_response(
    insp: DefectInspection,
    detections_list: list[DefectDetection] | None = None,
) -> InspectionResponse:
    dets = detections_list if detections_list is not None else insp.detections
    det_responses = []
    classes_found = set()
    for d in dets:
        classes_found.add(d.class_name)
        det_responses.append(DetectionResponse(
            id=d.id,
            detection_order=d.detection_order,
            class_id=d.class_id,
            class_name=d.class_name,
            confidence=round(d.confidence, 6),
            bbox=BboxResponse(
                x_min=round(d.bbox_x_min),
                y_min=round(d.bbox_y_min),
                x_max=round(d.bbox_x_max),
                y_max=round(d.bbox_y_max),
            ),
            pre_nms_index=d.pre_nms_index,
        ))

    return InspectionResponse(
        id=insp.id,
        image_width=insp.image_width,
        image_height=insp.image_height,
        image_url=_artifact_url(insp.image_path),
        thumbnail_url=_artifact_url(insp.thumbnail_path) if insp.thumbnail_path else None,
        detected_at=insp.detected_at,
        inference_time_ms=insp.inference_time_ms,
        summary=InspectionSummary(
            total_defects=insp.total_defects,
            classes_found=sorted(classes_found),
        ),
        model=ModelIdentity(
            model_version=insp.model_version,
            model_weights_hash=insp.model_weights_hash,
            device=insp.device,
            confidence_threshold=insp.confidence_threshold,
            iou_threshold=insp.iou_threshold,
            imgsz=insp.imgsz,
            experiment_id=resolve_inspection_experiment_id(
                insp.model_weights_hash, settings.C1_MODEL_FREEZE_MANIFEST,
            ),
        ),
        detections=det_responses,
        complexity_tier=insp.complexity_tier,
        complexity_score=insp.complexity_score,
    )


@router.get("/health", response_model=HealthResponse)
async def health():
    """Model health check with identity and version info."""
    info = detection_service.get_health()
    if info["status"] != "healthy":
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unavailable", "detail": info.get("detail", "Model not loaded")},
        )
    return info


@router.get("/readiness", response_model=ReadinessResponse)
async def readiness():
    """Readiness probe — is the service ready to accept inference requests?"""
    loaded = detection_service.is_loaded
    return ReadinessResponse(
        ready=loaded,
        model_loaded=loaded,
        detail="" if loaded else "Model not loaded",
    )


VALID_CLASS_NAMES = {"baekra", "color issues", "contamination", "cut", "gray stitch", "selvet", "stain"}


@router.get(
    "/inspections",
    response_model=InspectionListResponse,
    responses={400: {"model": ErrorResponse}},
)
async def list_inspections(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
    class_name: str | None = Query(default=None),
    from_date: datetime | None = Query(default=None),
    to_date: datetime | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
):
    """List inspections with optional filters and pagination."""
    if class_name and class_name not in VALID_CLASS_NAMES:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": f"Invalid class_name. Valid: {sorted(VALID_CLASS_NAMES)}"},
        )
    if from_date and to_date and from_date > to_date:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": "from_date must be before to_date"},
        )

    inspections, total = await repo.list_inspections(
        db, skip=skip, limit=limit,
        class_name=class_name, from_date=from_date, to_date=to_date,
    )

    items = []
    for insp in inspections:
        classes = sorted({d.class_name for d in insp.detections})
        items.append(InspectionListItem(
            id=insp.id,
            detected_at=insp.detected_at,
            thumbnail_url=_artifact_url(insp.thumbnail_path) if insp.thumbnail_path else None,
            total_defects=insp.total_defects,
            defect_classes=classes,
            model_version=insp.model_version,
        ))

    return InspectionListResponse(total=total, skip=skip, limit=limit, items=items)


@router.post(
    "/inspect",
    response_model=InspectionResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def inspect(
    file: UploadFile,
    confidence_threshold: float = Query(
        default=0.25, ge=0.0, le=1.0, description="Confidence threshold [0,1]"
    ),
    db: AsyncSession = Depends(get_db),
):
    """Upload an image and run defect detection."""
    if not detection_service.is_loaded:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "Model unavailable"},
        )

    raw_bytes = await file.read()

    try:
        content_type, image_bgr = validate_image_bytes(raw_bytes)
    except ImageValidationError as e:
        return JSONResponse(status_code=e.status_code, content={"detail": e.message})

    inspection_id = uuid.uuid4()
    storage = _get_storage()
    image_hash = compute_hash(raw_bytes)
    h, w = image_bgr.shape[:2]

    try:
        image_path = storage.save_original(inspection_id, raw_bytes, content_type)
        thumbnail_path = storage.save_thumbnail(inspection_id, image_bgr)
    except OSError as e:
        logger.error("Storage write failed for %s: %s", inspection_id, e)
        storage.cleanup_inspection(inspection_id)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Storage error"},
        )

    try:
        result, inference_time_ms = await detection_service.run_detection(
            image_bgr,
            conf_threshold=confidence_threshold,
            iou_threshold=0.5,
        )
    except RuntimeError:
        storage.cleanup_inspection(inspection_id)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "Model unavailable"},
        )
    except Exception as e:
        logger.error("Inference failed for %s: %s", inspection_id, e)
        storage.cleanup_inspection(inspection_id)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Inference error"},
        )

    if result.error_message:
        storage.cleanup_inspection(inspection_id)
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": result.error_message},
        )

    inspection = DefectInspection(
        id=inspection_id,
        image_path=image_path,
        thumbnail_path=thumbnail_path,
        image_width=w,
        image_height=h,
        image_hash=image_hash,
        detected_at=datetime.now(timezone.utc),
        model_version=detection_service.model_version,
        model_weights_hash=detection_service.weights_hash,
        total_defects=len(result.detections),
        confidence_threshold=confidence_threshold,
        iou_threshold=0.5,
        imgsz=640,
        inference_time_ms=inference_time_ms,
        device=detection_service.device,
    )

    detections = []
    for det in result.detections:
        detections.append(DefectDetection(
            detection_order=det.detection_id,
            class_id=det.class_id,
            class_name=det.class_name,
            confidence=det.confidence,
            bbox_x_min=det.bbox_xyxy[0],
            bbox_y_min=det.bbox_xyxy[1],
            bbox_x_max=det.bbox_xyxy[2],
            bbox_y_max=det.bbox_xyxy[3],
            pre_nms_index=det.pre_nms_index,
        ))

    try:
        await repo.create_inspection(db, inspection, detections)
        await db.commit()
    except Exception as e:
        logger.error("Database commit failed for %s: %s", inspection_id, e)
        await db.rollback()
        storage.cleanup_inspection(inspection_id)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Database error"},
        )

    return _build_inspection_response(
        inspection, detections_list=detections,
    )


@router.get(
    "/inspections/{inspection_id}",
    response_model=InspectionResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_inspection(
    inspection_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Retrieve a previously saved inspection by ID."""
    insp = await repo.get_inspection_by_id(db, inspection_id)
    if insp is None:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": "Inspection not found"},
        )
    return _build_inspection_response(insp)


METHOD_MAP = {"grad-cam": "gradcam", "grad-cam++": "gradcampp"}


def _build_explanation_response(expl: DefectExplanation) -> ExplanationResponse:
    return ExplanationResponse(
        id=expl.id,
        detection_id=expl.detection_id,
        xai_method=expl.xai_method,
        status=expl.status,
        outcome=expl.outcome,
        overlay_url=_artifact_url(expl.overlay_path) if expl.overlay_path else None,
        raw_heatmap_url=_artifact_url(expl.raw_heatmap_path) if expl.raw_heatmap_path else None,
        model_version=expl.model_version,
        target_layer=expl.target_layer,
        settings=expl.settings,
        error_message=expl.error_message,
        inference_time_ms=expl.inference_time_ms,
        generated_at=expl.generated_at,
        created_at=expl.created_at,
    )


@router.post(
    "/inspections/{inspection_id}/explanations",
    response_model=ExplanationResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def create_explanation(
    inspection_id: uuid.UUID,
    body: ExplanationRequest,
    db: AsyncSession = Depends(get_db),
):
    """Generate a Grad-CAM explanation for a detection."""
    if not detection_service.is_loaded:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "Model unavailable"},
        )

    det = await repo.get_detection_with_inspection(db, inspection_id, body.detection_id)
    if det is None:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": "Detection not found in this inspection"},
        )

    insp = await repo.get_inspection_for_detection(db, body.detection_id)
    if insp is None:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": "Inspection not found"},
        )

    if insp.model_version != detection_service.model_version:
        expl = DefectExplanation(
            detection_id=det.id,
            xai_method=body.method,
            status="completed",
            outcome="unavailable",
            model_version=detection_service.model_version or "",
            error_message=(
                f"Model mismatch: inspection used {insp.model_version}, "
                f"loaded model is {detection_service.model_version}"
            ),
            generated_at=datetime.now(timezone.utc),
        )
        db.add(expl)
        await db.commit()
        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content=_build_explanation_response(expl).model_dump(mode="json"),
        )

    storage = _get_storage()
    raw_bytes = storage.load_original_bytes(inspection_id)
    if raw_bytes is None:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": "Original image not found in storage"},
        )

    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": "Stored image could not be decoded"},
        )

    from app.components.component1.ml import Detection as MLDetection

    ml_detection = MLDetection(
        detection_id=det.detection_order,
        class_id=det.class_id,
        class_name=det.class_name,
        confidence=det.confidence,
        bbox_xyxy=[det.bbox_x_min, det.bbox_y_min, det.bbox_x_max, det.bbox_y_max],
        pre_nms_index=det.pre_nms_index,
    )

    internal_method = METHOD_MAP.get(body.method, "gradcam")

    explanation_id = uuid.uuid4()
    try:
        xai_result, elapsed_ms = await detection_service.run_explanation(
            image_bgr,
            ml_detection,
            method=internal_method,
            conf_threshold=insp.confidence_threshold,
            iou_threshold=insp.iou_threshold,
        )
    except Exception as e:
        logger.error("XAI failed for detection %s: %s", det.id, e)
        expl = DefectExplanation(
            id=explanation_id,
            detection_id=det.id,
            xai_method=body.method,
            status="failed",
            outcome="error",
            model_version=detection_service.model_version or "",
            error_message=str(e),
            generated_at=datetime.now(timezone.utc),
        )
        db.add(expl)
        await db.commit()
        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content=_build_explanation_response(expl).model_dump(mode="json"),
        )

    overlay_path = None
    raw_heatmap_path = None

    if xai_result.heatmap_overlay is not None:
        try:
            overlay_path = storage.save_overlay(inspection_id, explanation_id, xai_result.heatmap_overlay)
        except OSError as e:
            logger.error("Failed to save overlay for %s: %s", explanation_id, e)

    if xai_result.heatmap_raw is not None:
        try:
            raw_heatmap_path = storage.save_raw_heatmap(inspection_id, explanation_id, xai_result.heatmap_raw)
        except OSError as e:
            logger.error("Failed to save raw heatmap for %s: %s", explanation_id, e)

    xai_settings = {
        "pre_nms_index": det.pre_nms_index,
        "verify_score_atol": 0.0001,
        "method": internal_method,
        "conf_threshold": insp.confidence_threshold,
        "iou_threshold": insp.iou_threshold,
        "imgsz": insp.imgsz,
        "model_version": detection_service.model_version,
        "model_weights_hash": detection_service.weights_hash,
    }
    if xai_result.target_scale >= 0:
        xai_settings["scale"] = xai_result.target_scale

    expl = DefectExplanation(
        id=explanation_id,
        detection_id=det.id,
        xai_method=body.method,
        status="completed",
        outcome=xai_result.status,
        overlay_path=overlay_path,
        raw_heatmap_path=raw_heatmap_path,
        model_version=detection_service.model_version or "",
        target_layer=xai_result.target_layer,
        settings=xai_settings,
        error_message=xai_result.error_message or None,
        inference_time_ms=elapsed_ms,
        generated_at=datetime.now(timezone.utc),
    )

    try:
        db.add(expl)
        await db.commit()
    except Exception as e:
        logger.error("DB commit failed for explanation %s: %s", explanation_id, e)
        await db.rollback()
        storage.remove_explanation_artifacts(inspection_id, explanation_id)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Database error"},
        )

    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content=_build_explanation_response(expl).model_dump(mode="json"),
    )


@router.get(
    "/inspections/{inspection_id}/explanations/{explanation_id}",
    response_model=ExplanationResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_explanation(
    inspection_id: uuid.UUID,
    explanation_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Retrieve a saved explanation by ID."""
    expl = await repo.get_explanation_by_id(db, explanation_id)
    if expl is None:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": "Explanation not found"},
        )
    return _build_explanation_response(expl)


@router.get(
    "/inspections/{inspection_id}/detections/{detection_id}/explanations",
    response_model=list[ExplanationResponse],
)
async def list_explanations(
    inspection_id: uuid.UUID,
    detection_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """List all explanations for a detection."""
    det = await repo.get_detection_with_inspection(db, inspection_id, detection_id)
    if det is None:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": "Detection not found in this inspection"},
        )
    expls = await repo.get_explanations_for_detection(db, detection_id)
    return [_build_explanation_response(e) for e in expls]


@router.get("/artifacts/{path:path}")
async def serve_artifact(path: str):
    """Serve stored image artifacts. Local demo only — production uses object storage."""
    storage = _get_storage()
    full = storage.resolve_artifact_path(path)
    if full is None:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": "Artifact not found"},
        )
    suffix = full.suffix.lower()
    media = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".npy": "application/octet-stream"}
    return FileResponse(full, media_type=media.get(suffix, "application/octet-stream"))
