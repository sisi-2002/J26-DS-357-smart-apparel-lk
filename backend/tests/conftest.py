"""Global test-database isolation and fail-closed safety guard.

Root cause this fixes: tests that use the real `app.main.app` (test_inspect,
test_health, test_explanations, test_history, test_integration) trigger the
real FastAPI lifespan on `with TestClient(app) as c:`, which calls
`init_engine()` reading `settings.DATABASE_URL`. `settings` is a module-level
singleton (`app/core/config.py: settings = Settings()`) constructed once at
import time from `.env` — which points at the shared local dev Postgres
database (`smart_apparel_dev`). No test ever overrode this, so every HTTP
test that commits an inspection wrote a real row into that shared database.

Fix: for every test in this directory, before the test body runs we
(1) prepare a fresh, disposable, file-based SQLite database under pytest's
own tmp directory with the Component1 schema already created, (2) point
`settings.DATABASE_URL` at it, and (3) wrap `init_engine()` so that no
matter what later code does, it refuses to create an engine against
anything that isn't that exact pre-approved disposable SQLite file —
closed by default, not by trusting a database name.
"""
import asyncio
import os
import sys
import tempfile
import uuid

import pytest
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import app.db.session as db_session_module
from app.core.config import settings
from app.db.session import Base

# Base.metadata only contains tables for ORM classes that have actually
# been imported somewhere. Without this, whether `_strip_schemas()` below
# captures anything (and whether create_all() creates any tables at all)
# depends on which other test module happened to import these classes
# first — which is why this bug only reproduced when test_history.py was
# run alone, not as part of the full suite. Import explicitly so table
# registration never depends on collection order.
import app.components.component1.models  # noqa: F401


class UnsafeTestDatabaseError(RuntimeError):
    """Raised when a test would connect to a non-isolated database."""


def _assert_safe_test_database_url(url: str) -> None:
    """Allow-list check, not a name-prefix check.

    Safe iff: the SQLAlchemy backend is literally sqlite (a different
    driver/engine entirely from the Postgres dev/prod databases — renaming
    a Postgres database to something that "sounds disposable" can never
    satisfy this), AND it names a concrete file path that sits under the
    OS temp directory (never ':memory:', never empty, never a path outside
    temp). This is checked again, independently, every single time
    init_engine() is called — not just once at fixture setup.
    """
    if not url:
        raise UnsafeTestDatabaseError("Refusing to connect: DATABASE_URL is empty")

    parsed = make_url(url)
    if parsed.get_backend_name() != "sqlite":
        raise UnsafeTestDatabaseError(
            f"Refusing to run DB-writing tests against backend "
            f"{parsed.get_backend_name()!r}. Component1 tests must use an "
            f"isolated disposable SQLite database — never the configured "
            f"application DATABASE_URL, which may point at a shared, "
            f"development, or production database."
        )

    database = parsed.database
    if not database or database == ":memory:":
        raise UnsafeTestDatabaseError(
            "Refusing anonymous/in-memory SQLite URL — tests require an "
            "explicit, pre-verified disposable file path."
        )

    normalized = os.path.normpath(database)
    tmp_root = os.path.normpath(tempfile.gettempdir())
    if not normalized.startswith(tmp_root):
        raise UnsafeTestDatabaseError(
            f"Refusing SQLite path outside the system temp directory: {database}"
        )


def _strip_schemas() -> dict:
    original = {}
    for t in Base.metadata.sorted_tables:
        original[t.fullname] = t.schema
        t.schema = None
    return original


def _restore_schemas(original: dict) -> None:
    for t in Base.metadata.sorted_tables:
        if t.fullname in original:
            t.schema = original[t.fullname]


async def _create_schema(test_db_url: str) -> None:
    setup_engine = create_async_engine(test_db_url, echo=False)
    async with setup_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await setup_engine.dispose()


async def _dispose_lingering_engine() -> None:
    if db_session_module.engine is not None:
        await db_session_module.engine.dispose()
    db_session_module.engine = None
    db_session_module.async_session_factory = None


@pytest.fixture(autouse=True)
def _isolated_test_database(tmp_path, monkeypatch):
    """Runs for every test in this directory, before the test body.

    Autouse fixtures execute before explicitly-requested fixtures of the
    same scope (e.g. the `mock_service`/`client_with_model`-style fixtures
    in test_inspect.py/test_health.py that build `TestClient(app)`), so
    DATABASE_URL and the init_engine() guard are both in place before the
    FastAPI lifespan can ever run.

    This is a plain sync fixture (using asyncio.run() internally for the
    two DB setup/teardown calls) rather than a pytest_asyncio fixture,
    deliberately: none of the real-app HTTP tests in this directory use
    pytest_asyncio fixtures or async test functions — Starlette's
    TestClient drives the app's own event loop internally in a separate
    thread. Mixing an async conftest fixture with that pattern produced
    an observed event-loop-scope hazard (schema/tables created in one
    fixture-setup event loop were invisible to queries issued from the
    TestClient's own loop) that was timing/collection-order dependent —
    it passed when the full suite ran, but failed deterministically when
    a single test file was run alone. A synchronous fixture removes any
    interaction with pytest-asyncio's loop-scope handling entirely.
    """
    db_path = tmp_path / f"component1_test_{uuid.uuid4().hex}.sqlite3"
    test_db_url = f"sqlite+aiosqlite:///{db_path}"

    # Build and verify the URL against the same guard used at connection
    # time, so a bug in this fixture itself cannot silently produce an
    # unsafe default.
    _assert_safe_test_database_url(test_db_url)

    # Schema stripping mutates Base.metadata globally (SQLAlchemy Table
    # objects, not per-engine state) — it must stay stripped for the
    # entire time ANY engine (ours, or the real app's via init_engine())
    # might query these tables, matching the existing eval-test pattern.
    # Restoring it right after create_all() would make later queries use
    # schema-qualified names that don't exist in this unqualified SQLite
    # file ("no such table: component1.x").
    original_schemas = _strip_schemas()
    asyncio.run(_create_schema(test_db_url))

    monkeypatch.setattr(settings, "DATABASE_URL", test_db_url)
    # Settings is constructed at import time: setenv alone does not isolate
    # artifact writes. Keep test images/XAI beside the disposable database.
    monkeypatch.setattr(settings, "C1_STORAGE_DIR", str(tmp_path / "artifacts"))
    import app.components.component1.endpoints as endpoints
    monkeypatch.setattr(endpoints, "_storage", None)

    real_init_engine = db_session_module.init_engine

    def _guarded_init_engine():
        _assert_safe_test_database_url(settings.DATABASE_URL)
        real_init_engine()

    monkeypatch.setattr(db_session_module, "init_engine", _guarded_init_engine)

    try:
        yield test_db_url
    finally:
        # Defensive teardown even if a test skipped its own TestClient
        # context exit (e.g. raised before __exit__ ran dispose_engine()).
        asyncio.run(_dispose_lingering_engine())
        _restore_schemas(original_schemas)
