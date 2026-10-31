"""Alembic migration environment.

Reads DATABASE_URL from backend/.env via app.core.config. Converts the
asyncpg URL to a synchronous psycopg2 URL for migration execution.
Autogeneration is scoped to the 'component1' schema so tables owned by
other components or PostgreSQL internals are never touched.
"""
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Import models so Base.metadata knows about them
from app.core.config import settings  # noqa: E402
from app.db.session import Base  # noqa: E402
import app.components.component1.models  # noqa: E402, F401

target_metadata = Base.metadata

COMPONENT1_SCHEMA = "component1"


def _get_sync_url() -> str:
    """Convert postgresql+asyncpg:// → postgresql:// for Alembic sync driver."""
    url = settings.DATABASE_URL
    if not url:
        raise RuntimeError("DATABASE_URL not set — cannot run migrations")
    return url.replace("postgresql+asyncpg://", "postgresql+psycopg2://", 1)


def include_object(object, name, type_, reflected, compare_to):
    """Only include objects in the component1 schema."""
    if type_ == "table":
        return object.schema == COMPONENT1_SCHEMA
    if hasattr(object, "table") and hasattr(object.table, "schema"):
        return object.table.schema == COMPONENT1_SCHEMA
    return True


def run_migrations_offline() -> None:
    context.configure(
        url=_get_sync_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_schemas=True,
        include_object=include_object,
        version_table_schema=COMPONENT1_SCHEMA,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = _get_sync_url()
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_schemas=True,
            include_object=include_object,
            version_table_schema=COMPONENT1_SCHEMA,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
