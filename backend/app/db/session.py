"""Async SQLAlchemy engine, session factory and declarative base."""
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings

engine = None
async_session_factory = None


class Base(DeclarativeBase):
    pass


def init_engine():
    global engine, async_session_factory
    if not settings.DATABASE_URL:
        return
    engine = create_async_engine(
        settings.DATABASE_URL,
        echo=False,
        pool_pre_ping=True,
    )
    async_session_factory = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False,
    )


async def dispose_engine():
    global engine
    if engine is not None:
        await engine.dispose()
        engine = None


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    if async_session_factory is None:
        raise RuntimeError("Database not initialised — DATABASE_URL missing or init_engine() not called")
    async with async_session_factory() as session:
        yield session
