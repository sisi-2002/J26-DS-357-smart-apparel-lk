"""Component 1 database CRUD operations."""
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import distinct, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.components.component1.models import (
    DefectDetection,
    DefectExplanation,
    DefectInspection,
)


async def create_inspection(
    session: AsyncSession,
    inspection: DefectInspection,
    detections: list[DefectDetection],
) -> DefectInspection:
    session.add(inspection)
    for det in detections:
        det.inspection_id = inspection.id
        session.add(det)
    await session.flush()
    return inspection


async def get_inspection_by_id(
    session: AsyncSession,
    inspection_id: uuid.UUID,
) -> DefectInspection | None:
    stmt = (
        select(DefectInspection)
        .options(selectinload(DefectInspection.detections))
        .where(DefectInspection.id == inspection_id)
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def get_detection_with_inspection(
    session: AsyncSession,
    inspection_id: uuid.UUID,
    detection_id: uuid.UUID,
) -> DefectDetection | None:
    stmt = (
        select(DefectDetection)
        .join(DefectInspection)
        .where(
            DefectDetection.id == detection_id,
            DefectInspection.id == inspection_id,
        )
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def get_inspection_for_detection(
    session: AsyncSession,
    detection_id: uuid.UUID,
) -> DefectInspection | None:
    stmt = (
        select(DefectInspection)
        .join(DefectDetection)
        .where(DefectDetection.id == detection_id)
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def get_explanation_by_id(
    session: AsyncSession,
    explanation_id: uuid.UUID,
) -> DefectExplanation | None:
    stmt = select(DefectExplanation).where(DefectExplanation.id == explanation_id)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def get_explanations_for_detection(
    session: AsyncSession,
    detection_id: uuid.UUID,
) -> list[DefectExplanation]:
    stmt = (
        select(DefectExplanation)
        .where(DefectExplanation.detection_id == detection_id)
        .order_by(DefectExplanation.created_at)
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def list_inspections(
    session: AsyncSession,
    skip: int = 0,
    limit: int = 20,
    class_name: Optional[str] = None,
    from_date: Optional[datetime] = None,
    to_date: Optional[datetime] = None,
) -> tuple[list[DefectInspection], int]:
    base = select(DefectInspection)
    count_base = select(func.count(distinct(DefectInspection.id)))

    if class_name:
        base = base.join(DefectDetection).where(DefectDetection.class_name == class_name)
        count_base = count_base.join(DefectDetection).where(DefectDetection.class_name == class_name)
        base = base.distinct()

    if from_date:
        base = base.where(DefectInspection.detected_at >= from_date)
        count_base = count_base.where(DefectInspection.detected_at >= from_date)
    if to_date:
        base = base.where(DefectInspection.detected_at <= to_date)
        count_base = count_base.where(DefectInspection.detected_at <= to_date)

    total_result = await session.execute(count_base)
    total = total_result.scalar() or 0

    stmt = (
        base
        .options(selectinload(DefectInspection.detections))
        .order_by(DefectInspection.detected_at.desc(), DefectInspection.id)
        .offset(skip)
        .limit(limit)
    )
    result = await session.execute(stmt)
    return list(result.scalars().unique().all()), total
