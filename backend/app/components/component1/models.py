"""Component 1 ORM models.

Tables:
  defect_inspections, defect_detections, defect_explanations — live inference
  evaluation_runs, evaluation_samples, evaluation_detection_metrics — offline XAI evaluation

All tables live in the 'component1' PostgreSQL schema. Other components must
access this data through the API, never by direct table queries.
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Double,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base

SCHEMA = "component1"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class DefectInspection(Base):
    __tablename__ = "defect_inspections"
    __table_args__ = (
        Index("ix_defect_inspections_detected_at", "detected_at"),
        Index("ix_defect_inspections_complexity_tier", "complexity_tier"),
        Index("ix_defect_inspections_created_by", "created_by"),
        {"schema": SCHEMA},
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    image_path: Mapped[str] = mapped_column(String(512), nullable=False)
    thumbnail_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    image_width: Mapped[int] = mapped_column(Integer, nullable=False)
    image_height: Mapped[int] = mapped_column(Integer, nullable=False)
    image_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    detected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    model_weights_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)
    complexity_tier: Mapped[str | None] = mapped_column(String(16), nullable=True)
    complexity_score: Mapped[float | None] = mapped_column(Double, nullable=True)
    total_defects: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    confidence_threshold: Mapped[float] = mapped_column(Double, nullable=False, default=0.25)
    iou_threshold: Mapped[float] = mapped_column(Double, nullable=False, default=0.5)
    imgsz: Mapped[int] = mapped_column(Integer, nullable=False, default=640)
    inference_time_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    device: Mapped[str | None] = mapped_column(String(16), nullable=True)
    created_by: Mapped[uuid.UUID | None] = mapped_column(Uuid, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)

    detections: Mapped[list["DefectDetection"]] = relationship(
        back_populates="inspection", cascade="all, delete-orphan",
        order_by="DefectDetection.detection_order",
    )


class DefectDetection(Base):
    __tablename__ = "defect_detections"
    __table_args__ = (
        Index("ix_defect_detections_inspection_id", "inspection_id"),
        Index("ix_defect_detections_class_id", "class_id"),
        CheckConstraint("class_id BETWEEN 0 AND 6", name="chk_class_id"),
        CheckConstraint("confidence BETWEEN 0 AND 1", name="chk_confidence"),
        {"schema": SCHEMA},
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    inspection_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey(f"{SCHEMA}.defect_inspections.id", ondelete="CASCADE"), nullable=False,
    )
    detection_order: Mapped[int] = mapped_column(Integer, nullable=False)
    class_id: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    class_name: Mapped[str] = mapped_column(String(32), nullable=False)
    confidence: Mapped[float] = mapped_column(Double, nullable=False)
    bbox_x_min: Mapped[float] = mapped_column(Double, nullable=False)
    bbox_y_min: Mapped[float] = mapped_column(Double, nullable=False)
    bbox_x_max: Mapped[float] = mapped_column(Double, nullable=False)
    bbox_y_max: Mapped[float] = mapped_column(Double, nullable=False)
    pre_nms_index: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)

    inspection: Mapped["DefectInspection"] = relationship(back_populates="detections")
    explanations: Mapped[list["DefectExplanation"]] = relationship(
        back_populates="detection", cascade="all, delete-orphan",
    )


class DefectExplanation(Base):
    __tablename__ = "defect_explanations"
    __table_args__ = (
        Index("ix_defect_explanations_detection_id", "detection_id"),
        CheckConstraint(
            "status IN ('pending','processing','completed','failed')",
            name="chk_explanation_status",
        ),
        CheckConstraint(
            "outcome IS NULL OR outcome IN ('success','degenerate','unavailable','ambiguous','error')",
            name="chk_explanation_outcome",
        ),
        {"schema": SCHEMA},
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    detection_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey(f"{SCHEMA}.defect_detections.id", ondelete="CASCADE"), nullable=False,
    )
    xai_method: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    outcome: Mapped[str | None] = mapped_column(String(16), nullable=True)
    overlay_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    raw_heatmap_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    target_layer: Mapped[str | None] = mapped_column(String(128), nullable=True)
    settings: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    inference_time_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    generated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)

    detection: Mapped["DefectDetection"] = relationship(back_populates="explanations")


# ── Offline XAI evaluation tables ────────────────────────────


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"
    __table_args__ = (
        Index("ix_evaluation_runs_created_at", "created_at"),
        Index("ix_evaluation_runs_status", "status"),
        CheckConstraint(
            "status IN ('registered','running','completed','failed','cancelled')",
            name="chk_eval_run_status",
        ),
        CheckConstraint(
            "calibration_status IN ('none','pending_qc','qc_validated')",
            name="chk_eval_calibration_status",
        ),
        CheckConstraint(
            "selection_mode IN ('primary_representatives','all_views')",
            name="chk_eval_selection_mode",
        ),
        {"schema": SCHEMA},
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    run_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    registry_version: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="registered")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    notes: Mapped[str] = mapped_column(Text, nullable=False, default="")

    # Protocol settings (frozen per run)
    protocol_version: Mapped[str] = mapped_column(String(16), nullable=False)
    conf_threshold: Mapped[float] = mapped_column(Double, nullable=False)
    iou_threshold_inference: Mapped[float] = mapped_column(Double, nullable=False)
    imgsz: Mapped[int] = mapped_column(Integer, nullable=False)
    iou_threshold_matching: Mapped[float] = mapped_column(Double, nullable=False)
    require_class_match: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    xai_methods: Mapped[list] = mapped_column(JSONB, nullable=False)
    seed: Mapped[int] = mapped_column(Integer, nullable=False, default=42)

    # Manifest identity
    manifest_id: Mapped[str] = mapped_column(String(64), nullable=False)
    source_manifest_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    split_plan_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    selected_sample_list_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    split: Mapped[str] = mapped_column(String(8), nullable=False)
    selection_mode: Mapped[str] = mapped_column(String(32), nullable=False)
    sample_count: Mapped[int] = mapped_column(Integer, nullable=False)

    # Model identity
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    model_weights_hash: Mapped[str] = mapped_column(String(128), nullable=False)

    # Calibration
    calibration_status: Mapped[str] = mapped_column(String(16), nullable=False, default="none")
    calibration_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    calibration_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)

    # Package versions and full config snapshot
    package_versions: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    config_snapshot: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Progress counters
    completed_samples: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failed_samples: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_samples: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    samples: Mapped[list["EvaluationSample"]] = relationship(
        back_populates="run", cascade="all, delete-orphan",
    )


class EvaluationSample(Base):
    __tablename__ = "evaluation_samples"
    __table_args__ = (
        UniqueConstraint("run_id", "sample_id", name="uq_eval_sample_run_sample"),
        Index("ix_evaluation_samples_run_id", "run_id"),
        Index("ix_evaluation_samples_source_unit_id", "source_unit_id"),
        Index("ix_evaluation_samples_assignment_group_root", "assignment_group_root"),
        CheckConstraint(
            "status IN ('completed','failed','skipped')",
            name="chk_eval_sample_status",
        ),
        {"schema": SCHEMA},
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey(f"{SCHEMA}.evaluation_runs.id", ondelete="CASCADE"), nullable=False,
    )
    sample_id: Mapped[str] = mapped_column(String(256), nullable=False)
    filename: Mapped[str] = mapped_column(String(256), nullable=False)
    source_key: Mapped[str] = mapped_column(String(256), nullable=False)
    is_augmented: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    source_unit_id: Mapped[str] = mapped_column(String(256), nullable=False)
    confirmed_family_id: Mapped[str] = mapped_column(String(256), nullable=False, default="")
    assignment_group_root: Mapped[str] = mapped_column(String(256), nullable=False)
    label_status: Mapped[str] = mapped_column(String(16), nullable=False)
    annotation_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    error_message: Mapped[str] = mapped_column(Text, nullable=False, default="")

    # Detection match counts
    detection_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tp_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    fp_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    fn_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Raw complexity features (null if unavailable)
    complexity_luminance_var: Mapped[float | None] = mapped_column(Double, nullable=True)
    complexity_glcm_entropy: Mapped[float | None] = mapped_column(Double, nullable=True)
    complexity_colour_var: Mapped[float | None] = mapped_column(Double, nullable=True)

    # Timing
    detection_time_ms: Mapped[float | None] = mapped_column(Double, nullable=True)
    explanation_time_ms: Mapped[float | None] = mapped_column(Double, nullable=True)
    total_time_ms: Mapped[float | None] = mapped_column(Double, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)

    run: Mapped["EvaluationRun"] = relationship(back_populates="samples")
    detection_metrics: Mapped[list["EvaluationDetectionMetric"]] = relationship(
        back_populates="sample", cascade="all, delete-orphan",
    )


class EvaluationDetectionMetric(Base):
    __tablename__ = "evaluation_detection_metrics"
    __table_args__ = (
        UniqueConstraint(
            "sample_id", "detection_order", "xai_method",
            name="uq_eval_det_metric_sample_det_method",
        ),
        Index("ix_eval_det_metrics_sample_id", "sample_id"),
        Index("ix_eval_det_metrics_class_id", "detection_class_id"),
        CheckConstraint(
            "match_type IN ('TP','FP','')",
            name="chk_eval_match_type",
        ),
        CheckConstraint(
            "xai_outcome IN ('success','degenerate','error','unavailable','ambiguous')",
            name="chk_eval_xai_outcome",
        ),
        {"schema": SCHEMA},
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    sample_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey(f"{SCHEMA}.evaluation_samples.id", ondelete="CASCADE"), nullable=False,
    )

    # Detection identity
    detection_order: Mapped[int] = mapped_column(Integer, nullable=False)
    detection_class_id: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    detection_class_name: Mapped[str] = mapped_column(String(32), nullable=False)
    detection_confidence: Mapped[float] = mapped_column(Double, nullable=False)
    detection_bbox_x_min: Mapped[float] = mapped_column(Double, nullable=False)
    detection_bbox_y_min: Mapped[float] = mapped_column(Double, nullable=False)
    detection_bbox_x_max: Mapped[float] = mapped_column(Double, nullable=False)
    detection_bbox_y_max: Mapped[float] = mapped_column(Double, nullable=False)
    detection_bbox_area: Mapped[float] = mapped_column(Double, nullable=False)

    # GT matching
    match_type: Mapped[str] = mapped_column(String(4), nullable=False)
    match_iou: Mapped[float | None] = mapped_column(Double, nullable=True)
    matched_gt_idx: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # XAI method and outcome
    xai_method: Mapped[str] = mapped_column(String(16), nullable=False)
    xai_outcome: Mapped[str] = mapped_column(String(16), nullable=False)
    xai_error: Mapped[str] = mapped_column(Text, nullable=False, default="")

    # Box-based localization metrics (null when not applicable, NOT zero)
    pointing_game: Mapped[float | None] = mapped_column(Double, nullable=True)
    energy_concentration: Mapped[float | None] = mapped_column(Double, nullable=True)

    # Region-restricted faithfulness (null when not applicable)
    insertion_auc: Mapped[float | None] = mapped_column(Double, nullable=True)
    deletion_auc: Mapped[float | None] = mapped_column(Double, nullable=True)
    insertion_random_auc: Mapped[float | None] = mapped_column(Double, nullable=True)

    # Method agreement — paired-method result, stored once per detection pair
    # NOT duplicated: only the row for the first xai_method gets the value;
    # query via detection_order grouping, not per-method duplication
    method_agreement_ssim: Mapped[float | None] = mapped_column(Double, nullable=True)

    # Heatmap diagnostics
    heatmap_nonzero_fraction: Mapped[float | None] = mapped_column(Double, nullable=True)
    heatmap_entropy: Mapped[float | None] = mapped_column(Double, nullable=True)

    # Timing
    generation_time_ms: Mapped[float | None] = mapped_column(Double, nullable=True)

    # Skip/error reasons (JSONB — metric_name → reason string)
    metric_skip_reasons: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Artifact references (heatmap file paths within run directory)
    heatmap_artifact_path: Mapped[str | None] = mapped_column(String(512), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)

    sample: Mapped["EvaluationSample"] = relationship(back_populates="detection_metrics")
