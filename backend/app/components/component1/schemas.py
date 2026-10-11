"""Pydantic schemas for Component 1 API responses."""
import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    model_version: Optional[str] = None
    ultralytics_version: Optional[str] = None
    torch_version: Optional[str] = None
    device: Optional[str] = None
    classes: Optional[list[str]] = None
    loaded_at: Optional[datetime] = None
    experiment_id: Optional[str] = None
    checkpoint_verified: Optional[bool] = None


class ReadinessResponse(BaseModel):
    ready: bool
    model_loaded: bool
    detail: str = ""


class BboxResponse(BaseModel):
    x_min: int
    y_min: int
    x_max: int
    y_max: int


class DetectionResponse(BaseModel):
    id: uuid.UUID
    detection_order: int
    class_id: int
    class_name: str
    confidence: float
    bbox: BboxResponse
    pre_nms_index: int


class InspectionSummary(BaseModel):
    total_defects: int
    classes_found: list[str]


class ModelIdentity(BaseModel):
    model_version: str
    model_weights_hash: Optional[str] = None
    device: Optional[str] = None
    confidence_threshold: float
    iou_threshold: float
    imgsz: int
    experiment_id: Optional[str] = None


class InspectionResponse(BaseModel):
    id: uuid.UUID
    image_width: int
    image_height: int
    image_url: str
    thumbnail_url: Optional[str] = None
    detected_at: datetime
    inference_time_ms: Optional[int] = None
    summary: InspectionSummary
    model: ModelIdentity
    detections: list[DetectionResponse]
    complexity_tier: Optional[str] = None
    complexity_score: Optional[float] = None


class ExplanationRequest(BaseModel):
    detection_id: uuid.UUID
    method: str = Field(default="grad-cam", pattern=r"^(grad-cam|grad-cam\+\+)$")


class ExplanationResponse(BaseModel):
    id: uuid.UUID
    detection_id: uuid.UUID
    xai_method: str
    status: str
    outcome: Optional[str] = None
    overlay_url: Optional[str] = None
    raw_heatmap_url: Optional[str] = None
    model_version: str
    target_layer: Optional[str] = None
    settings: Optional[dict] = None
    error_message: Optional[str] = None
    inference_time_ms: Optional[int] = None
    generated_at: Optional[datetime] = None
    created_at: datetime


class InspectionListItem(BaseModel):
    id: uuid.UUID
    detected_at: datetime
    thumbnail_url: Optional[str] = None
    total_defects: int
    defect_classes: list[str]
    model_version: str


class InspectionListResponse(BaseModel):
    total: int
    skip: int
    limit: int
    items: list[InspectionListItem]


class ErrorResponse(BaseModel):
    detail: str
